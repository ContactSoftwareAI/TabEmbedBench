import numpy as np
import polars as pl
from sklearn.base import TransformerMixin
from tabembedbench.sphere_model import SphereModelARF as SphereModel

from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator


class SphereBasedEmbedding(AbstractEmbeddingGenerator):
    """Sphere-based embedding generator for tabular data.

    This embedding model projects tabular data onto high-dimensional spheres,
    treating categorical and numerical features differently. Categorical features
    are embedded as points in small regions around random sphere points, while
    numerical features are embedded along radial directions.

    Attributes:
        embed_dim (int): Dimensionality of the embedding space.
        categorical_indices (list[int] | None): Indices of categorical columns.
        column_properties (list): List storing embedding properties for each column.
        n_cols (int | None): Number of columns in the fitted data.
    """

    def __init__(
        self, embed_dim: int
    ) -> None:
        """Initialize the sphere-based embedding generator.

        Args:
            embed_dim (int): Dimensionality of the embedding space.
        """
        super().__init__(name=f"Sphere-Based (Dim {embed_dim})")
        self.categorical_indices = None
        self.embed_dim = embed_dim
        self.column_properties = []
        self.n_cols = None
        self.model = SphereModel(embed_dim=embed_dim)

    def _preprocess_data(
        self, X: np.ndarray, train: bool = True, outlier: bool = False, **kwargs
    ) -> np.ndarray:
        """Preprocess input data (no-op for SphereBasedEmbedding).

        Args:
            X (np.ndarray): Input data to preprocess.
            train (bool, optional): Whether this is training mode. Defaults to True.
            outlier (bool, optional): Whether to handle outliers. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: The input data unchanged.
        """
        return X

    def _fit_model(
        self,
        data: np.ndarray,
        categorical_indices: list[int] | None = None,
        **kwargs,
    ):
        """Fit the embedding model to preprocessed data.

        Args:
            data (np.ndarray): Preprocessed input data.
            categorical_indices (list[int] | None, optional): Categorical column indices.
                Defaults to None.
            **kwargs: Additional keyword arguments (unused).
        """
        if categorical_indices is None:
            categorical_indices = []
            #categorical_indices = infer_categorical_columns(data)
        self.model.fit(data, categorical_indices=categorical_indices)

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray,
        X_test_preprocessed: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute embeddings for the input data.

        Args:
            data (np.ndarray): Input data to embed.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Embeddings of shape (n_samples, embed_dim).
        """
        embeddings_train = self.model.transform(X_train_preprocessed)
        embeddings_test = None if X_test_preprocessed is None else self.model.transform(X_test_preprocessed)

        return embeddings_train, embeddings_test

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Clears all fitted column properties and metadata.
        """
        self.model = SphereModel(self.embed_dim)
