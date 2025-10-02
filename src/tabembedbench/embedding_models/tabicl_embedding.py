import inspect
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted
from tabicl.model.embedding import ColEmbedding
from tabicl.model.inference_config import InferenceConfig
from tabicl.model.interaction import RowInteraction
from tabicl.sklearn.preprocessing import (
    CustomStandardScaler,
    PreprocessingPipeline,
    RTDLQuantileTransformer,
)
from torch import nn

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device


class TabICLRowEmbedding(nn.Module):
    """A neural network module for embedding tabular data using column and
    row interactions.

    This class is designed to process tabular data by embedding columnar and
    row-wise information. It combines column embeddings with row-level
    interaction mechanisms, making it suitable for tasks such as tabular
    data modeling and representation learning. The model leverages hierarchical
    attention mechanisms on both column and row levels to extract meaningful
    embeddings.

    Attributes:
        col_embedder (ColEmbedding): Embedding mechanism for columns, which
            processes column-level features to generate column representations.
        row_interactor (RowInteraction): Interaction mechanism for rows that
            processes column representations to extract row-level embeddings.

    References:
    [1] Qu, J. et al. (2025). Tabicl: A tabular foundation model for in-context
        learning on large data. arXiv preprint arXiv:2502.05564.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: Union[str, callable] = "gelu",
        norm_first: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            reserve_cls_tokens=row_num_cls,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

    def forward(
        self,
        X: torch.Tensor,
        feature_shuffles: list[list[int]] | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> torch.Tensor:
        if inference_config is None:
            inference_config = InferenceConfig()

        column_representations = self.col_embedder(
            X, feature_shuffles=feature_shuffles, mgr_config=inference_config.COL_CONFIG
        )

        return self.row_interactor(
            column_representations, mgr_config=inference_config.ROW_CONFIG
        )


class TabICLEmbedding(AbstractEmbeddingGenerator):
    """Generates embeddings from TabICL models with preprocessing capabilities and
    optional embedding normalization.

    The TabICLEmbedding class is designed to create embeddings using
    pre-trained TabICL models. It supports preprocessing for input data and
    provides an option for embedding normalization. The class also allows
    working with pretrained TabICL model checkpoints or downloading them
    automatically if not provided. It encapsulates preprocessing pipelines for
    outlier and non-outlier data handling, and provides functionalities for
    resetting and configuring the preprocess pipelines.

    Attributes:
        model_path (Optional[Path]): Path to the TabICL model checkpoint;
            downloads from HF Hub if not provided or path doesn't exist.
        tabicl_row_embedder (TabICLRowEmbedding): The row embedding model is
            loaded with specific weights and configuration.
        normalize_embeddings (bool): Specifies whether generated embeddings
            should be normalized.
        preprocess_pipeline (PreprocessingPipeline): Pipeline for preprocessing
            input data.
        outlier_preprocessing_pipeline (OutlierPreprocessingPipeline):
            Dedicated pipeline for preprocessing outlier data.
    """

    def __init__(
        self,
        model_path: str | None = None,
        normalize_embeddings: bool = False,
        preprocess_tabicl_data: bool = False,
    ):
        """Initializes the TabICLEmbedding class, handling configuration and setup
        for the tabular data embedder. Optionally loads a specific model
        path and applies settings for embedding normalization and data
        preprocessing.

        Args:
            model_path: Optional path to the model file as a string. If None, no model
                is loaded.
            normalize_embeddings: Boolean flag indicating whether to normalize
                embeddings or not.
            preprocess_tabicl_data: Boolean flag to determine whether to preprocess
                Tabicl data.
        """
        super().__init__(name="tabicl-classifier-v1.1-0506")

        self.model_path = Path(model_path) if model_path is not None else None
        self.tabicl_row_embedder = self.get_tabicl_model()

        self.normalize_embeddings = normalize_embeddings
        self._preprocess_tabicl_data = preprocess_tabicl_data
        self.preprocess_pipeline = PreprocessingPipeline()
        self.outlier_preprocessing_pipeline = OutlierPreprocessingPipeline()

    def get_tabicl_model(self):
        if self.model_path is not None and self.model_path.exists():
            model_ckpt_path = self.model_path
        else:
            model_ckpt_path = hf_hub_download(
                repo_id="jingang/TabICL-clf",
                filename="tabicl-classifier-v1.1-0506.ckpt",
            )

        model_ckpt = torch.load(model_ckpt_path)

        state_dict = model_ckpt["state_dict"]
        config = model_ckpt["config"]

        filtered_config = filter_params_for_class(TabICLRowEmbedding, config)

        row_embedding_model = TabICLRowEmbedding(
            **filtered_config,
        )

        row_embedding_model.load_state_dict(state_dict, strict=False)

        for param in row_embedding_model.col_embedder.parameters():
            param.requires_grad = False

        for param in row_embedding_model.row_interactor.parameters():
            param.requires_grad = False

        row_embedding_model.eval()

        return row_embedding_model

    def _preprocess_data(
        self,
        X: np.ndarray,
        train: bool = True,
        outlier: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses the input data based on the specified mode (training or inference) and
        whether the data contains outliers.

        This function applies preprocessing pipelines to the input data depending on the training
        or inference mode and handles outlier data if specified. If training mode is enabled, the
        method fits and transforms the data using the appropriate preprocessing pipeline, either
        for outlier or standard data. During inference or non-training mode, it only transforms
        the data using the respective pipeline.

        Args:
            X: Input data to be preprocessed, passed as a numpy array.
            train: Boolean flag indicating whether the data is in training mode. If True, the
                method fits and transforms the data; otherwise, it only performs transformation.
            outlier: Boolean flag indicating if the input data contains outliers. If True,
                the outlier-specific preprocessing pipeline is used; otherwise, the standard
                preprocessing pipeline is applied.

        Returns:
            np.ndarray: Preprocessed data after applying the corresponding preprocessing steps.
        """
        X_preprocess = X

        if train and self._preprocess_tabicl_data:
            if outlier:
                X_preprocess = self.outlier_preprocessing_pipeline.fit_transform(
                    X_preprocess
                )
            else:
                X_preprocess = self.preprocess_pipeline.fit_transform(X_preprocess)
        else:
            if self._preprocess_tabicl_data:
                if outlier:
                    X_preprocess = self.outlier_preprocessing_pipeline.transform(
                        X_preprocess
                    )
                else:
                    X_preprocess = self.preprocess_pipeline.transform(X_preprocess)

        return X_preprocess

    def _fit_model(self, X_preprocessed: np.ndarray, train: bool = True,
                   **kwargs):
        return X_preprocessed

    def _compute_embeddings(
        self,
        X: np.ndarray,
        device: torch.device | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Computes the embeddings for the given input data.

        The method processes 2D or 3D input arrays and computes embeddings using
        a row embedder. The input tensor is sent to the specified device, converted
        to float type, and processed through the forward method of the embedder.
        The resulting embeddings are returned as a NumPy array.

        Args:
            X (np.ndarray): The input data for which embeddings are to be computed.
                Must be a 2D or 3D array.
            device (torch.device | None): The device on which to perform the
                computation. If None, the default device is determined using
                `get_device()`.

        Returns:
            np.ndarray: The computed embeddings as a NumPy array.
        """
        if device is None:
            device = get_device()
            self.tabicl_row_embedder.to(device)

        if len(X.shape) not in [2, 3]:
            raise ValueError("Input must be 2D or 3D array")

        X = torch.from_numpy(X).float().to(device)
        if len(X.shape) == 2:
            X = X.unsqueeze(0)

        return self.tabicl_row_embedder.forward(X).cpu().squeeze().numpy()

    def _reset_embedding_model(self):
        """
        Resets the embedding model by reinitializing preprocessing pipelines.

        This method reinitializes both the primary preprocessing pipeline and the
        outlier-specific preprocessing pipeline to their original states. It is
        useful when you need to reset the state of the preprocessing components
        for a fresh evaluation or after model updates.

        """
        self.preprocess_pipeline = PreprocessingPipeline()
        self.outlier_preprocessing_pipeline = OutlierPreprocessingPipeline()


def filter_params_for_class(cls, params_dict):
    sig = inspect.signature(cls.__init__)

    valid_params = set(sig.parameters.keys()) - {"self"}

    return {k: v for k, v in params_dict.items() if k in valid_params}


# The code is taken from the original TabICL repo, only the
# OutlierRemover is removed. The rest is similar to the original code.
class OutlierPreprocessingPipeline(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        normalization_method: str = "power",
        outlier_threshold: float = 4.0,
        random_state: int | None = None,
    ):
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self._validate_data(X)

        # 1. Apply standard scaling
        self.standard_scaler_ = CustomStandardScaler()
        X_scaled = self.standard_scaler_.fit_transform(X)

        # 2. Apply normalization
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(
                    method="yeo-johnson", standardize=True
                )
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state
                )
            elif self.normalization_method == "quantile_rtdl":
                self.normalizer_ = Pipeline(
                    [
                        (
                            "quantile_rtdl",
                            RTDLQuantileTransformer(
                                output_distribution="normal",
                                random_state=self.random_state,
                            ),
                        ),
                        ("std", StandardScaler()),
                    ]
                )
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.normalization_method}"
                )

            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None

        return self

    def transform(self, X):
        """Apply the preprocessing pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns:
        -------
        X_out : ndarray
            Preprocessed data.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, copy=True)
        # Standard scaling
        X = self.standard_scaler_.transform(X)
        # Normalization
        if self.normalizer_ is not None:
            try:
                # this can fail in rare cases if there is an outlier in X
                # that was not present in fit()
                X = self.normalizer_.transform(X)
            except ValueError:
                # clip values to train min/max
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)

        return X
