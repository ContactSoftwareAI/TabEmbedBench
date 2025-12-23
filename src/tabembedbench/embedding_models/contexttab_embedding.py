from huggingface_hub import hf_hub_download
import numpy as np
from typing import Tuple
from pathlib import Path
import torch
import gc
import os
import pandas as pd
from pandas import DataFrame, Series
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device
from sap_rpt_oss.model.torch_model import RPT
from sap_rpt_oss.constants import ModelSize
from sap_rpt_oss.data.tokenizer import Tokenizer



class ConTextTabModel(RPT):

    def __init__(self, model_size: ModelSize, regression_type: str, classification_type: str, num_regression_bins: int = 1):
        super().__init__(model_size,
                         regression_type=regression_type,
                         classification_type=classification_type
        )

    ## aus contexttab repo
    def forward(self, data: dict[str, torch.Tensor], labels=None, **kwargs):
        input_embeds = self.embeddings(data, is_regression=True)
        # (max_num_rows, max_num_columns, hidden_size)

        extended_attention_mask = self.build_context_attention_mask(data, input_embeds.device)
        extended_attention_mask = extended_attention_mask.type(input_embeds.dtype)

        # ToDo: If im original beachten und evtl. anpassen
        for layer in self.in_context_encoder:
            input_embeds = layer(input_embeds, extended_attention_mask)
        encoder_outputs = input_embeds

        # encoder_outputs has shape (num_rows, num_columns, hidden_size)
        target_column_output = encoder_outputs[:, -1]  # (num_rows, hidden_size)

        return target_column_output


class ConTextTabEmbedding(AbstractEmbeddingGenerator):
    def __init__(self, device: str | None = None):
        super().__init__(name="ConTextTab")
        self.device = device if device is not None else get_device()
        self.MAX_NUM_COLUMNS = 500
        self.max_context_size = 8192
        self.seed = 42
        self.regression_type = "l2"
        self.classification_type = "cross-entropy"
        self.num_regression_bins = 1

        self.contexttab_embedder = self._get_model().to(self.device)
        self.tokenizer = Tokenizer(
            regression_type=self.regression_type,
            classification_type=self.classification_type,
            random_seed=self.seed,
            num_regression_bins=self.num_regression_bins,
            is_valid=True)


    def _get_model(self):
        model_size = ModelSize.base
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        checkpoint_path = hf_hub_download(
            repo_id="SAP/sap-rpt-1-oss",
            filename='2025-11-04_sap-rpt-one-oss.pt',
            token=hf_token
        )
        # Der Checkpoint scheint für einfache Regression (1 Bin) ausgelegt zu sein
        model = ConTextTabModel(model_size, regression_type=self.regression_type,
                             classification_type=self.classification_type,
                             num_regression_bins=self.num_regression_bins)
    

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()

        return model

    def _preprocess_data(self, X: pd.DataFrame, train: bool = True, outlier: bool = False,
                         **kwargs):
        #ToDo: Bagging?
        if not isinstance(X, DataFrame):
        #ToDo: Handle column names
            X = DataFrame(X)
        y = Series(np.zeros(len(X)), name="target", index=X.index)

        df = pd.concat([X, y.to_frame()], axis=1)

        # There is no bagging, but we still have to sample because there are too many points
        if len(df) > self.max_context_size:
            df = df.sample(self.max_context_size, replace=False, random_state=self.seed)

        # Remove constant columns
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        constant_cols = list(X.columns[X.nunique() == 1])
        if constant_cols:
            X = X.drop(columns=constant_cols)
            df = pd.concat([X, y], axis=1)

        # If number of columns exceed maximum
        if df.shape[1] > self.MAX_NUM_COLUMNS:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            X = X.sample(n=self.MAX_NUM_COLUMNS - 1, axis=1, random_state=self.seed, replace=False)
            df = pd.concat([X, y], axis=1)

        df = df.iloc[:len(df)]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]

        # The tokenizer fails if y_query is empty because StandardScaler requires at least one sample.
        # If we are just generating embeddings for the input X, we can use X itself as the query.
        X_query = X.copy() #pd.DataFrame(columns=X.columns)
        y_query = y.copy() #pd.DataFrame(columns=y.columns)

        data, labels, label_classes = self.tokenizer(X, y, X_query, y_query, "regression")
        # Pass the same series twice (y_train and y_test) to satisfy the API requirement
        _, target_mean, target_std = self.tokenizer.standard_scale_column(y, y)

        return {
            'data': data,
            'num_rows': df.shape[0],
            'num_cols': df.shape[1],
            'labels': None,
            'is_regression': torch.tensor(True),
            'label_classes': np.asarray(label_classes),
            'target_mean': target_mean,
            'target_std': target_std
        }

    def _fit_model(self, X_preprocessed: dict,
                   y_preprocessed: dict | None = None, train: bool = True,
                   **kwargs) -> None:
        self._is_fitted = True

    def _compute_embeddings(
            self,
            X_train_preprocessed: np.ndarray,
            X_test_preprocessed: np.ndarray | None = None,
            outlier: bool = False,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray | None]:

        if outlier:
            embeddings = self.contexttab_embedder(X_train_preprocessed)

            return embeddings, None

        else:
            embeddings_train = self.contexttab_embedder(X_train_preprocessed)
            X_train_test_stack = np.vstack([X_train_preprocessed, X_test_preprocessed])
            embeddings = self.contexttab_embedder(X_train_test_stack)
            embeddings_test = embeddings[len(X_train_preprocessed):]

            return embeddings_train, embeddings_test

    def _reset_embedding_model(self, *args, **kwargs):
        """
        Resets the embedding model by performing a series of steps to free up memory
        and ensure proper resource management. This includes moving the model to the CPU,
        clearing the preprocessing pipeline, releasing unused memory, and clearing the GPU
        or MPS cache depending on the device type. Finally, it relocates the model back to
        the specified device for future use.

        Parameters:
            args (Any): Additional positional arguments.
            kwargs (Any): Additional keyword arguments.
        """
        # Move model to CPU before deleting references
        if self.contexttab_embedder is not None:
            self.contexttab_embedder.cpu()

        # Clear preprocessing pipeline
        self.preprocess_pipeline = None
        #self._is_fitted = False

        # Force garbage collection
        gc.collect()

        # Clear GPU cache
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()
        elif self.device == "mps":
            import torch

            torch.mps.empty_cache()

        # Move model back to the device for next use
        if self.contexttab_embedder is not None:
            self.contexttab_embedder.to(self.device)