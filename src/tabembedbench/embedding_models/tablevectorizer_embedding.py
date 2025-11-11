import numpy as np
import polars as pl
from skrub import TableVectorizer

from tabembedbench.embedding_models import AbstractEmbeddingGenerator


class TableVectorizerEmbedding(AbstractEmbeddingGenerator):
    """TableVectorizer-based embedding generator for tabular data.

    This class uses skrub's TableVectorizer to transform mixed-type tabular data
    into numerical embeddings. TableVectorizer automatically handles different column
    types (numerical, categorical, datetime, text) and applies appropriate encoding
    strategies for each, making it a robust baseline for tabular data representation.

    The embedding generation process:
    1. Automatically detects column types in the input data
    2. Applies appropriate transformations (one-hot encoding, target encoding, etc.)
    3. Returns dense numerical representations suitable for downstream tasks

    Attributes:
        tablevectorizer (TableVectorizer): The underlying skrub TableVectorizer instance
            that handles the automatic feature transformation.
        _is_fitted (bool): Whether the model has been fitted to data.

    Example:
        >>> embedding_gen = TableVectorizerEmbedding()
        >>> train_emb, test_emb, time = embedding_gen.generate_embeddings(
        ...     X_train, X_test
        ... )
    """

    def __init__(self, **kwargs):
        """Initialize the TableVectorizer embedding generator.

        Args:
            **kwargs: Additional keyword arguments to pass to TableVectorizer,
                such as 'n_jobs' for parallel processing or specific encoder
                configurations.
        """
        super().__init__(name="TableVectorizer")

        self.tablevectorizer = TableVectorizer(**kwargs)
        self._is_fitted = False

    def _preprocess_data(
        self, X: np.ndarray, train: bool = True, outlier: bool = False, **kwargs
    ) -> pl.DataFrame:
        """Preprocess input data (no-op for TableVectorizer).

        TableVectorizer handles preprocessing internally, so this method
        simply returns the input data unchanged.

        Args:
            X (np.ndarray): Input data to preprocess.
            train (bool, optional): Whether this is training mode. Defaults to True.
            outlier (bool, optional): Whether to handle outliers. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: The input data unchanged.
        """
        X = pl.from_numpy(X)

        return X

    def _fit_model(
        self,
        X_preprocessed: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit the TableVectorizer to the input data.

        This method fits the TableVectorizer, which learns the appropriate
        transformations for each column type in the data.

        Args:
            X_preprocessed (np.ndarray): Preprocessed input data.
            **kwargs: Additional keyword arguments (unused).
        """
        self.tablevectorizer.fit(X_preprocessed)
        self._is_fitted = True

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray,
        X_test_preprocessed: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute embeddings using the fitted TableVectorizer.

        Transforms the input data into dense numerical representations using
        the fitted TableVectorizer. Handles both outlier mode (train only) and
        standard mode (train + test).

        Args:
            X_train_preprocessed (np.ndarray): Preprocessed training data.
            X_test_preprocessed (np.ndarray | None, optional): Preprocessed test data.
                Required when outlier is False. Defaults to None.
            outlier (bool, optional): If True, transforms only training data.
                Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            tuple[np.ndarray, np.ndarray | None]: A tuple containing:
                - train_embeddings: Transformed training data
                - test_embeddings: Transformed test data, or None if outlier is True.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if outlier:
            embeddings = self.tablevectorizer.transform(X_train_preprocessed)
            return embeddings.to_numpy(), None
        if self._is_fitted:
            embeddings_train = self.tablevectorizer.transform(X_train_preprocessed)
            embeddings_test = self.tablevectorizer.transform(X_test_preprocessed)

            return embeddings_train.to_numpy(), embeddings_test.to_numpy()
        else:
            raise ValueError("Model is not fitted.")

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Reinitializes the TableVectorizer and clears fitted state.
        """
        self.tablevectorizer = TableVectorizer()
        self._is_fitted = False
