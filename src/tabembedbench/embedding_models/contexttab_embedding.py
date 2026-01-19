from huggingface_hub import hf_hub_download
import numpy as np
from typing import Tuple, Union, Optional
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

## aus ConTextTab Repo
def to_device(x, device: Union[torch.device, int], dtype: Optional[torch.dtype] = None, raise_on_unexpected=True):
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            target_dtype = dtype if v.dtype == torch.float32 else v.dtype
            x[k] = v.to(device, dtype=target_dtype)
        elif isinstance(v, dict):
            x[k] = to_device(v, device, dtype=dtype)
        elif v is not None and raise_on_unexpected:
            raise ValueError(f'Unknown type, {type(v)}')
    return x

class ConTextTabModel(RPT):
    """
    Represents a model that extends RPT for contextual tabular data processing.

    This class is designed to handle regression and classification tasks using a transformer-based
    framework. It processes tabular data by encoding it into contextual embeddings and applying
    attention mechanisms through its in-context encoder layers. It produces a representation of
    the target column for downstream tasks like regression or classification.

    Attributes:
        model_size: Specifies the size configuration of the model.
        regression_type: Specifies the type of regression task being performed.
        classification_type: Specifies the type of classification task being performed.
        num_regression_bins: The number of bins for discretizing regression labels. Default is 1.
    """
    def __init__(self, model_size: ModelSize, regression_type: str, classification_type: str, num_regression_bins: int = 1):
        super().__init__(model_size,
                         regression_type=regression_type,
                         classification_type=classification_type
        )


    ## aus contexttab repo
    def forward(self, data: dict[str, torch.Tensor], labels=None, **kwargs):
        """
        Processes input data through the forward pass of the model, applying embeddings,
        in-context encoder layers, and attention mechanisms. Returns the outputs corresponding
        to the target column.

        Parameters:
            data (dict[str, torch.Tensor]): Input data containing tensors required by the model.
            labels (optional): Optional labels for supervision during training, if applicable.
            **kwargs: Additional keyword arguments that might be needed during forward pass.

        Returns:
            torch.Tensor: Output tensor of shape (num_rows, hidden_size), corresponding to the
            processed representation of the target column.
        """
        data = to_device(data, self.device, raise_on_unexpected=False)
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
    """
    Represents an embedding generator using the ConTextTab architecture.

    This class is designed to generate embeddings for structured tabular data using the ConTextTab
    model. It supports regression and classification tasks, while also providing preprocessing
    capabilities to handle large datasets, constant columns, and feature selection. Additionally, it
    includes integration with tokenizer components and model checkpoints for efficient embedding
    generation and task handling.

    Attributes:
        device (str): The device on which the model operates (e.g., 'cuda' or 'cpu'). Defaults to
            the device returned by the `get_device` function if not provided.
        MAX_NUM_COLUMNS (int): The maximum number of columns allowed for input data. Any additional
            columns are dropped during preprocessing.
        max_context_size (int): The maximum number of samples allowed in the context.
        seed (int): The random seed for reproducibility during sampling and preprocessing.
        regression_type (str): The regression type used in the model ('l2' by default).
        classification_type (str): The classification type used in the model ('cross-entropy' by default).
        num_regression_bins (int): The number of bins for regression tasks. Defaults to 1.

    Methods:
        _preprocess_data(X: pd.DataFrame, train: bool = True, outlier: bool = False, **kwargs)
            Preprocess data for modeling and tokenization, handling constraints on data size,
            constant columns, and sampling.

        _compute_embeddings(X_train_preprocessed: dict, X_test_preprocessed: dict | None = None,
                            outlier: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray | None]
            Compute embeddings for train and test datasets using the contexttab embedder.

        _fit_model(X_preprocessed: dict, y_preprocessed: dict | None = None, train: bool = True,
                   **kwargs) -> None
            Fit the model using preprocessed input data. Marks the model as fitted.

    """
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
        """
        Preprocess data for modeling and tokenization.

        This method takes an input DataFrame, optionally processes it for training or outlier
        detection, and prepares it for subsequent tokenization. The preprocessing includes
        handling of constant columns, column count restrictions, sampling large DataFrames,
        and generating necessary embeddings.

        Parameters:
            X (pd.DataFrame): The input data for preprocessing.
            train (bool): A flag indicating whether the preprocessing is to be applied
                for training purposes. Defaults to True.
            outlier (bool): A flag indicating whether the preprocessing is aimed at
                outlier detection. Defaults to False.
            **kwargs: Additional arguments that may be passed to the method.

        Returns:
            dict: A dictionary containing the processed data and other properties such as
                number of query samples.

        Raises:
            TypeError: If `X` is not of type `pd.DataFrame`.

        """
        #ToDo: Bagging?
        if not isinstance(X, DataFrame):
        #ToDo: Handle column names
            X = DataFrame(X)

        rng = np.random.default_rng(self.seed)
        y = Series(rng.standard_normal(len(X)), name="target", index=X.index)

        df = pd.concat([X, y.to_frame()], axis=1)

        # There is no bagging, but we still have to sample because there are too many points
        # Limit datasets to 8192 samples (context size) or include bagging
        # if len(df) > self.max_context_size:
        #    df = df.sample(self.max_context_size, replace=False, random_state=self.seed)

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
        X_query = X.copy()
        y_query = y.copy()

        data, labels, label_classes = self.tokenizer(X, y, X_query, y_query, "regression")
        # Pass the same series twice (y_train and y_test) to satisfy the API requirement
        data['num_query_samples'] = len(X_query)

        return data

    def _fit_model(self, X_preprocessed: dict,
                   y_preprocessed: dict | None = None, train: bool = True,
                   **kwargs) -> None:
        self._is_fitted = True

    def _compute_embeddings(
            self,
            X_train_preprocessed: dict,
            X_test_preprocessed: dict | None = None,
            outlier: bool = False,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Compute embeddings for train and test datasets using the contexttab embedder.

        This method processes preprocessed input data and computes embeddings using a
        pre-trained model. It can also handle separate outlier detection or combined
        train-test embeddings based on the `outlier` flag.

        Parameters:
        X_train_preprocessed : dict
            A dictionary of preprocessed training inputs. Tensor values in the dictionary
            are transferred to the appropriate device (e.g., GPU or CPU) before use.
        X_test_preprocessed : dict | None, optional
            A dictionary of preprocessed testing inputs. Tensor values in the dictionary
            are transferred to the appropriate device (e.g., GPU or CPU) before use. If None,
            only training embeddings are computed.
        outlier : bool, optional
            A flag indicating whether to compute outlier embeddings for the training data.
        **kwargs
            Additional keyword arguments for embedding computation.

        Returns:
        Tuple[np.ndarray, np.ndarray | None]
            A tuple containing two numpy arrays:
            - The first array contains embeddings for the training data.
            - The second array contains embeddings for the testing data or None if `X_test_preprocessed` is not provided.
        """
        self.contexttab_embedder.eval()
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'

        X_train_preprocessed = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for k, v in X_train_preprocessed.items()}
        if X_test_preprocessed is not None:
            X_test_preprocessed = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in X_test_preprocessed.items()}

        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=(device_type == 'cuda')):
            if outlier:
                embeddings = self.contexttab_embedder(X_train_preprocessed)
                # Only return query embeddings (last N)
                n_query = X_train_preprocessed['num_query_samples']
                return embeddings[-n_query:].cpu().numpy(), None

            else:
                res_train = self.contexttab_embedder(X_train_preprocessed)
                n_query_train = X_train_preprocessed['num_query_samples']
                n_query_test = X_test_preprocessed['num_query_samples']
                embeddings_train = res_train[-n_query_train:].cpu().numpy()

                combined_data = {}
                for key in X_train_preprocessed.keys():
                    if key == 'column_embeddings':
                        combined_data[key] = X_train_preprocessed[key]
                    elif isinstance(X_train_preprocessed[key], torch.Tensor) and X_test_preprocessed is not None:
                        # Concatenate context and query parts
                        ctx_len = X_train_preprocessed[key].shape[0] - n_query_train
                        context_part = X_train_preprocessed[key][:ctx_len]
                        query_part = X_test_preprocessed[key][-n_query_test:]
                        combined_data[key] = torch.cat([context_part, query_part], dim=0)
                    else:
                        combined_data[key] = X_train_preprocessed[key]

                res_test = self.contexttab_embedder(combined_data)
                embeddings_test = res_test[-n_query_test:].cpu().numpy()

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