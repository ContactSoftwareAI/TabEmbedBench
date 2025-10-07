from typing import Optional

import numpy as np
import polars as pl
from skrub import TableVectorizer

from tabembedbench.embedding_models import AbstractEmbeddingGenerator


class TabVectorizerEmbedding(AbstractEmbeddingGenerator):
    """TableVectorizer-based embedding generator for tabular data.

    This embedding model uses skrub's TableVectorizer to transform tabular data
    into numerical embeddings. It provides a simple baseline approach for
    converting mixed-type tabular data into vector representations.

    Attributes:
        optimize (bool): Whether to optimize hyperparameters (not yet implemented).
        tablevectorizer (TableVectorizer): The underlying TableVectorizer instance.
        _is_fitted (bool): Whether the model has been fitted to data.
    """

    def __init__(self, optimize: bool = False, **kwargs):
        """Initialize the TableVectorizer embedding generator.

        Args:
            optimize (bool, optional): Whether to optimize hyperparameters.
                Currently not implemented. Defaults to False.
            **kwargs: Additional keyword arguments to pass to TableVectorizer.
        """
        super().__init__(name="TabVectorizerEmbedding")
        self.optimize = optimize

        self.tablevectorizer = TableVectorizer(**kwargs)

    def _preprocess_data(self, X, train=True, **kwargs):
        """Preprocess input data by converting to Polars DataFrame.

        Args:
            X (np.ndarray): Input data to preprocess.
            train (bool, optional): Whether this is training data. Defaults to True.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            pl.DataFrame: Input data converted to Polars DataFrame format.
        """
        X = pl.from_numpy(X)

        return X

    def _fit_model(
        self,
        X_preprocessed: np.ndarray,
        y_preprocessed: Optional[np.ndarray] = None,
        train: bool = True,
        **kwargs,
    ):
        """Fit the TableVectorizer to the preprocessed data.

        Args:
            X_preprocessed (np.ndarray): Preprocessed input data.
            y_preprocessed (np.ndarray | None, optional): Target values (unused).
                Defaults to None.
            train (bool, optional): Whether to fit the model. Defaults to True.
            **kwargs: Additional keyword arguments (unused).
        """
        if train:
            self.tablevectorizer.fit(X_preprocessed)
            self._is_fitted = True

    def _compute_embeddings(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Compute embeddings using the fitted TableVectorizer.

        Args:
            X (np.ndarray): Input data to transform into embeddings.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: The computed embeddings.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._is_fitted:
            embeddings = self.tablevectorizer.transform(X)

            return embeddings.to_numpy()
        else:
            raise ValueError("Model is not fitted.")

    def _optimize_tablevectorizer(self):
        """Optimize TableVectorizer hyperparameters (not yet implemented).

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Reinitializes the TableVectorizer and clears fitted state.
        """
        self.tablevectorizer = TableVectorizer()
        self._is_fitted = False
