from huggingface_hub import hf_hub_download
import numpy as np
from typing import Tuple, Union
import safetensors.torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from tabstar.arch.config import TabStarConfig, E5_SMALL
from tabstar.arch.fusion import NumericalFusion
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.arch.interaction import InteractionEncoder
from tabstar.training.dataloader import get_dataloader
import torch
import gc
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device
from tabstar.training.devices import clear_cuda_cache
from sap_rpt_oss.model.torch_model import RPT


class ConTextTabModel(PreTrainedModel):

    def __init__(self, config: TabStarConfig):
        super().__init__(config)
        self.model = RPT.from_pretrained("SAP/concontext-tab")


    def forward(self, x_txt: np.ndarray, x_num: np.ndarray, d_output: int) -> Tensor:
        textual_embeddings = self.get_textual_embedding(x_txt)
        if not isinstance(x_num, Tensor):
            x_num = torch.tensor(x_num, dtype=textual_embeddings.dtype, device=textual_embeddings.device)
        embeddings = self.numerical_fusion(textual_embeddings=textual_embeddings, x_num=x_num)
        encoded = self.tabular_encoder(embeddings)
        target_tokens = encoded[:, :d_output]
        return target_tokens

    def get_embedding(self, x_txt: np.array, text_batch_size: int) -> Tensor:
        return embeddings


class ConTextTabEmbedding(AbstractEmbeddingGenerator):
    def __init__(self, device: str | None = None):
        """
        Initializes an instance of the class with specified attributes.

        Attributes:
        device (str | None): Specifies the device to be used. Defaults to the output of
            the get_device() method if not provided.
        preprocess_pipeline: Placeholder for preprocessing pipeline. This attribute
            is initially set to None.
        use_amp (bool): Indicates whether Automatic Mixed Precision (AMP) should be
            used based on the device type ('cuda' enables AMP).
        tabstar_row_embedder: The TabStar row embedding model instance, moved to
            the specified device.

        Args:
        device (str | None): The device to use, such as 'cuda' or 'cpu'.
        """
        super().__init__(name="TabStar")
        self.preprocess_pipeline = None
        self.device = device if device is not None else get_device()
        self.use_amp = bool(self.device.type == "cuda")
        self.tabstar_row_embedder = self._get_model().to(self.device)

    def _get_model(self):
        """
        Loads the TabStar embedding model and its configuration. The method handles
        downloading the model weights from the Hugging Face Hub and initializes the
        model with the relevant configuration.

        Returns:
            TabStarModel: An instance of the TabStar embedding model.

        Raises:
            OSError: If there is an issue while downloading or loading the weights.
        """
        model_ckpt_path = hf_hub_download(
            repo_id="alana89/TabSTAR",
            filename="model.safetensors"
        )
        config_class = TabStarConfig()
        tabstar_embedding_model = TabStarModel(config=config_class)
        weights = safetensors.torch.load_file(model_ckpt_path)
        tabstar_embedding_model.load_state_dict(weights, strict=False)
        tabstar_embedding_model.eval()

        return tabstar_embedding_model

    def _preprocess_data(self, X: np.ndarray, train: bool = True, outlier: bool = False,
                         **kwargs) -> TabSTARData:
        """
        Preprocesses the input data for the model. This includes handling outliers and
        filtering irrelevant data based on the configured preprocessing pipeline. If
        training data is provided, the pipeline is fitted; otherwise, the previously
        fitted pipeline is used to transform data.

        Parameters:
        X : np.ndarray
            Input feature data represented as a NumPy array.
        train : bool, optional
            Flag to indicate whether the data is for training. Defaults to True.
        outlier : bool, optional
            Flag to specify if outlier handling should be applied. Defaults to False.
        **kwargs :
            Additional keyword arguments for preprocessing.

        Raises:
        ValueError
            If `train` is False and the preprocessing pipeline is not yet fitted.

        Returns:
        TabSTARData
            Preprocessed data ready for model usage.
        """
        X = DataFrame(X)
        X.columns = [str(col) for col in X.columns]
        y = Series(np.zeros(X.shape[0]))
        if train:
            self.preprocess_pipeline = TabSTARVerbalizer(is_cls=False)
            self.preprocess_pipeline.fit(X,y)
            X_preprocessed = self.preprocess_pipeline.transform(X,y=None)
        else:
            if self.preprocess_pipeline is None:
                raise ValueError("Preprocessing pipeline is not fitted")
            else:
                X_preprocessed = self.preprocess_pipeline.transform(X,y=None)

        return X_preprocessed

    def _fit_model(self, X_preprocessed: np.ndarray,
                   y_preprocessed: np.ndarray | None = None, train: bool = True,
                   **kwargs) -> None:
        self._is_fitted = True

    def _compute_embeddings(
            self,
            X_train_preprocessed: np.ndarray,
            X_test_preprocessed: np.ndarray | None = None,
            outlier: bool = False,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Computes embeddings for preprocessed training and optionally testing data.

        This method is responsible for generating embeddings for given training and
        testing preprocessed datasets using a tabular embedding model. It optionally
        handles outlier detection mode and leverages GPU acceleration mechanisms if
        enabled. During training mode, embeddings for both training and testing data
        will be computed and returned. In outlier detection mode, only embeddings for
        training data are computed, returning solely this result with no test embeddings.
        The model must be fitted prior to invoking this method; otherwise, a ValueError
        will be raised.

        Parameters:
            X_train_preprocessed: np.ndarray
                Preprocessed training data.
            X_test_preprocessed: np.ndarray | None, default=None
                Preprocessed testing data. Can be None if outlier mode is enabled.
            outlier: bool, default=False
                Flag indicating whether the computation is for outlier detection.
            **kwargs
                Additional keyword arguments to pass to utility functions.

        Returns:
            Tuple[np.ndarray, np.ndarray | None]
                A tuple containing the training and testing data embeddings. Testing
                data embeddings will be None if `outlier` is True.

        Raises:
            ValueError:
                If the model has not been fitted before calling this method.
        """
        self.tabstar_row_embedder.eval()
        embeddings_list = []

        if outlier:
            dataloader = get_dataloader(X_train_preprocessed, is_train=False, batch_size=128)
            for data in dataloader:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    batch_predictions = self.tabstar_row_embedder(x_txt=data.x_txt, x_num=data.x_num, d_output=data.d_output)
#                    batch_predictions_numpy = batch_predictions.detach().cpu().squeeze().numpy()
                    batch_predictions_numpy = batch_predictions.detach().cpu().numpy()

                    if batch_predictions_numpy.ndim == 1:
                        batch_predictions_numpy = batch_predictions_numpy.reshape(1, -1)
                    elif batch_predictions_numpy.ndim > 2:
                        # Remove unnecessary dimensions but keep batch and feature dimensions
                        batch_predictions_numpy = batch_predictions_numpy.reshape(batch_predictions_numpy.shape[0], -1)

                    embeddings_list.append(batch_predictions_numpy)

            embeddings = np.concatenate(embeddings_list, axis=0)

            return embeddings, None

        if self._is_fitted:
            dataloader_train = get_dataloader(X_train_preprocessed, is_train=False, batch_size=128)
            for data in dataloader_train:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    batch_predictions = self.tabstar_row_embedder(x_txt=data.x_txt, x_num=data.x_num,
                                                                  d_output=data.d_output)
                    batch_predictions_numpy = batch_predictions.detach().cpu().squeeze().numpy()
                    embeddings_list.append(batch_predictions_numpy)

            embeddings_train = np.concatenate(embeddings_list, axis=0)

            embeddings_list = []
            dataloader_test = get_dataloader(X_test_preprocessed, is_train=False, batch_size=128)
            for data in dataloader_test:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    batch_predictions = self.tabstar_row_embedder(x_txt=data.x_txt, x_num=data.x_num,
                                                                  d_output=data.d_output)
                    batch_predictions_numpy = batch_predictions.detach().cpu().squeeze().numpy()
                    embeddings_list.append(batch_predictions_numpy)

            embeddings_test = np.concatenate(embeddings_list, axis=0)

            return embeddings_train, embeddings_test
        else:
            raise ValueError("Model is not fitted")


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
        if self.tabstar_row_embedder is not None:
            self.tabstar_row_embedder.cpu()

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
        if self.tabstar_row_embedder is not None:
            self.tabstar_row_embedder.to(self.device)