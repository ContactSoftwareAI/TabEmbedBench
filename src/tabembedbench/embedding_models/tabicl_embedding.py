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
    """TabICL-based embedding generator for tabular data.

    This class implements an embedding generator using the TabICL (Tabular In-Context
    Learning) foundation model. It leverages pre-trained transformer-based architectures
    with hierarchical attention mechanisms to generate contextual embeddings for
    tabular data. The model processes data through column embeddings and row interactions
    to capture both feature-level and sample-level relationships.

    The embedding generation process includes:
    1. Data preprocessing using TabICL-specific pipelines (standard or outlier-specific)
    2. Conversion to torch tensors and device placement
    3. Forward pass through the pre-trained TabICL model
    4. Extraction of row-level embeddings

    Attributes:
        model_path (Path | None): Path to a local model checkpoint. If None, downloads
            from HuggingFace hub.
        tabicl_row_embedder (TabICLRowEmbedding): The loaded TabICL model with frozen
            parameters.
        preprocess_pipeline (PreprocessingPipeline | OutlierPreprocessingPipeline | None):
            Fitted preprocessing pipeline for data transformation.
        device (str): Device for computation ('cuda' or 'cpu').

    References:
        [1] Qu, J. et al. (2025). Tabicl: A tabular foundation model for in-context
            learning on large data. arXiv preprint arXiv:2502.05564.

    Example:
        >>> embedding_gen = TabICLEmbedding()
        >>> train_emb, test_emb, time = embedding_gen.generate_embeddings(
        ...     X_train, X_test
        ... )
    """

    def __init__(self, model_path: str | None = None, device: str | None = None):
        """Initialize the TabICL embedding generator.

        Args:
            model_path (str | None, optional): Path to a local TabICL model checkpoint.
                If None, the model will be downloaded from HuggingFace. Defaults to None.
            device (str | None, optional): Device to use for computation ('cuda' or 'cpu').
                If None, automatically detects GPU availability. Defaults to None.
        """
        super().__init__(name="TabICL")

        self.model_path = Path(model_path) if model_path is not None else None
        self.tabicl_row_embedder = self.get_tabicl_model()
        self.preprocess_pipeline = None
        self.device = device if device is not None else get_device()

    def get_tabicl_model(self):
        """Load or download the TabICL model with pre-trained weights.

        This method either loads a model from a local checkpoint or downloads
        the pre-trained TabICL classifier from HuggingFace Hub. The model's
        parameters are frozen to prevent updates during embedding generation.

        Returns:
            TabICLRowEmbedding: The loaded TabICL row embedding model with
                pre-trained weights and frozen parameters for both column
                embedder and row interactor components.

        Note:
            The model is automatically set to evaluation mode with all gradients
            disabled to ensure consistent embedding generation.
        """
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
        self, X: np.ndarray, train: bool = True, outlier: bool = False, **kwargs
    ) -> np.ndarray:
        """Preprocess input data using TabICL-specific pipelines.

        Applies optional TableVectorizer preprocessing followed by either standard
        or outlier-specific preprocessing pipelines based on the data type.

        Args:
            X (np.ndarray): Input data to preprocess.
            train (bool, optional): Whether to fit the preprocessing pipelines.
                Defaults to True.
            outlier (bool, optional): Whether to use outlier-specific preprocessing.
                Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Preprocessed data ready for embedding computation.
        """
        if train:
            if outlier:
                self.preprocess_pipeline = OutlierPreprocessingPipeline()
                X_preprocessed = self.preprocess_pipeline.fit_transform(X)
            else:
                self.preprocess_pipeline = PreprocessingPipeline()
                X_preprocessed = self.preprocess_pipeline.fit_transform(X)
        else:
            if self.preprocess_pipeline is None:
                raise ValueError("Preprocessing pipeline is not fitted")
            else:
                X_preprocessed = self.preprocess_pipeline.transform(X)

        return X_preprocessed

    def _fit_model(
        self, X_preprocessed: np.ndarray, train: bool = True, **kwargs
    ) -> None:
        """Fit the model (no-op for TabICL as it uses pre-trained weights).

        Args:
            X_preprocessed (np.ndarray): Preprocessed input data.
            train (bool, optional): Whether this is training mode. Defaults to True.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: The preprocessed data unchanged.
        """
        self._is_fitted = True

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray,
        X_test_preprocessed: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Compute embeddings using the pre-trained TabICL model.

        This method generates embeddings by passing preprocessed data through the
        TabICL row embedder. In standard mode, it processes train and test data
        together to leverage TabICL's in-context learning capabilities. In outlier
        mode, only training data embeddings are computed.

        Args:
            X_train_preprocessed (np.ndarray): Preprocessed training dataset. Must be a 2D or
                3D NumPy array of shape (n_samples, n_features) or (1, n_samples, n_features).
            X_test_preprocessed (np.ndarray | None, optional): Preprocessed testing dataset.
                Must be a 2D or 3D NumPy array. Required when outlier is False. Defaults to None.
            outlier (bool, optional): If True, computes embeddings only for training data.
                If False, computes embeddings for both train and test by concatenating them
                to enable in-context learning. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            tuple[np.ndarray, np.ndarray | None]: A tuple containing:
                - train_embeddings: Embeddings for training data of shape (n_train, embed_dim)
                - test_embeddings: Embeddings for test data of shape (n_test, embed_dim),
                  or None if outlier is True.

        Raises:
            ValueError: If the input data is not a 2D or 3D array, or if the model
                is not fitted.
        """
        if len(X_train_preprocessed.shape) not in [2, 3]:
            raise ValueError("Input must be 2D or 3D array")

        if outlier:
            X_preprocessed = (
                torch.from_numpy(X_train_preprocessed).float().to(self.device)
            )
            if len(X_preprocessed.shape) == 2:
                X_preprocessed = X_preprocessed.unsqueeze(0)

            embeddings = (
                self.tabicl_row_embedder.forward(X_preprocessed).cpu().squeeze().numpy()
            )

            return embeddings, None

        if self._is_fitted:
            X_train_torch = (
                torch.from_numpy(X_train_preprocessed).float().to(self.device)
            )
            X_test_torch = torch.from_numpy(X_test_preprocessed).float().to(self.device)

            if len(X_train_torch.shape) == 2:
                X_train_torch = X_train_torch.unsqueeze(0)

            if len(X_test_torch.shape) == 2:
                X_test_torch = X_test_torch.unsqueeze(0)

            embeddings_train = (
                self.tabicl_row_embedder.forward(X_train_torch).cpu().squeeze().numpy()
            )

            size_X_train = X_train_torch.shape[1]

            X_train_test_stack = torch.cat((X_train_torch, X_test_torch), dim=1)

            X_embeddings = (
                self.tabicl_row_embedder.forward(X_train_test_stack)
                .cpu()
                .squeeze()
                .numpy()
            )

            embeddings_test = X_embeddings[size_X_train:]

            if embeddings_train.ndim == 1:
                embeddings_train = embeddings_train.reshape(1, -1)
            if embeddings_test.ndim == 1:
                embeddings_test = embeddings_test.reshape(1, -1)

            return embeddings_train, embeddings_test
        else:
            raise ValueError("Model is not fitted")

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Reinitializes all preprocessing pipelines to clear fitted state.
        """
        self.preprocess_pipeline = None
        self._is_fitted = False


def filter_params_for_class(cls, params_dict):
    """Filter parameters dictionary to only include valid parameters for a class.

    Args:
        cls: The class to filter parameters for.
        params_dict (dict): Dictionary of parameters to filter.

    Returns:
        dict: Filtered dictionary containing only valid parameters for the class.
    """
    sig = inspect.signature(cls.__init__)

    valid_params = set(sig.parameters.keys()) - {"self"}

    return {k: v for k, v in params_dict.items() if k in valid_params}


# The code is taken from the original TabICL repo, only the
# OutlierRemover is removed. The rest is similar to the original code.
class OutlierPreprocessingPipeline(TransformerMixin, BaseEstimator):
    """Preprocessing pipeline for outlier detection tasks.

    This pipeline applies standard scaling followed by optional normalization
    using various methods (power transform, quantile transform, etc.).

    Attributes:
        normalization_method (str): Method for normalization.
        outlier_threshold (float): Threshold for outlier detection.
        random_state (int | None): Random seed for reproducibility.
        standard_scaler_ (CustomStandardScaler): Fitted standard scaler.
        normalizer_ (Pipeline | None): Fitted normalization pipeline.
        X_min_ (np.ndarray): Minimum values for clipping.
        X_max_ (np.ndarray): Maximum values for clipping.
    """

    def __init__(
        self,
        normalization_method: str = "power",
        outlier_threshold: float = 4.0,
        random_state: int | None = None,
    ):
        """Initialize the outlier preprocessing pipeline.

        Args:
            normalization_method (str, optional): Normalization method to use.
                Options: 'power', 'quantile', 'quantile_rtdl', 'robust', 'none'.
                Defaults to "power".
            outlier_threshold (float, optional): Threshold for outlier detection.
                Defaults to 4.0.
            random_state (int | None, optional): Random seed for reproducibility.
                Defaults to None.
        """
        self.normalization_method = normalization_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the preprocessing pipeline to the data.

        Args:
            X: Input data to fit.
            y: Unused parameter, kept for sklearn compatibility. Defaults to None.

        Returns:
            self: The fitted pipeline.
        """
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
        """Apply the preprocessing pipeline to the data.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Preprocessed data.
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
