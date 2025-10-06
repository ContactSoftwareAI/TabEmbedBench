from typing import Optional

import numpy as np
import polars as pl
from skrub import TableVectorizer

from tabembedbench.embedding_models import AbstractEmbeddingGenerator


class TabVectorizerEmbedding(AbstractEmbeddingGenerator):
    """Handles the embedding generation using a TableVectorizer model.

    This class provides a wrapper for the TabVectorizer transformer from the
    skrub package. It represents the most basic transformations for tabular data
    to numerical values.

    Attributes:
        tablevectorizer (TableVectorizer): Instance of the TableVectorizer model
            used for generating row embeddings.
    """

    def __init__(self, optimize: bool = False, **kwargs):
        super().__init__(name="TabVectorizerEmbedding")
        self.optimize = optimize

        self.tablevectorizer = TableVectorizer(**kwargs)

    def _preprocess_data(self, X, train=True, **kwargs):
        """
        Preprocesses the input data by converting it from a NumPy array to a
        specific format and optionally fitting it using the model.

        Args:
            X: numpy.ndarray
                The input data to be processed.
            train: bool, optional
                A flag indicating if the model should be fitted with the input
                data (default is True).

        Returns:
            Any:
                The processed input data in the required format.
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
        if train:
            self.tablevectorizer.fit(X_preprocessed)
            self._is_fitted = True

    def _compute_embeddings(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes embeddings for the input data using the TableVectorizer transformation.

        This method processes the provided data with the TableVectorizer
        to generate vectorized numerical embeddings represented as a NumPy array.

        Args:
            X (np.ndarray): Input data to transform into embeddings.

        Returns:
            np.ndarray: The computed embeddings as a NumPy array.
        """
        if self._is_fitted:
            embeddings = self.tablevectorizer.transform(X)

            return embeddings.to_numpy()
        else:
            raise ValueError("Model is not fitted.")

    def _optimize_tablevectorizer(self):
        raise NotImplementedError

    def _reset_embedding_model(self):
        self.tablevectorizer = TableVectorizer()
        self._is_fitted = False
