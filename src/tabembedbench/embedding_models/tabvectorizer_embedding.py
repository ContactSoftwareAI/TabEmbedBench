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
        tablevectorizer (TableVectorizer): The underlying TableVectorizer instance.
        _is_fitted (bool): Whether the model has been fitted to data.
    """

    def __init__(self, **kwargs):
        """Initialize the TableVectorizer embedding generator.

        Args:
            optimize (bool, optional): Whether to optimize hyperparameters.
                Currently not implemented. Defaults to False.
            **kwargs: Additional keyword arguments to pass to TableVectorizer.
        """
        super().__init__(name="TableVectorizerEmbedding")

        self.tablevectorizer = TableVectorizer(**kwargs)
        self._is_fitted = False

    def _preprocess_data(
            self,
            X,
            train=True,
            outlier: bool = False,
            **kwargs
    ) -> np.ndarray:
        """
        Preprocesses the input data for use in model training or prediction. This
        method prepares the dataset by applying necessary transformations or filtering
        based on whether the data is considered part of the training phase or
        contains outliers.

        Args:
            X (np.ndarray): Input data to be processed.
            train (bool): Specifies whether the input data is in the training phase.
                Defaults to True.
            outlier (bool): Specifies whether to account for outliers during
                preprocessing. Defaults to False.
            **kwargs: Additional arguments that can be used for further customization
                during preprocessing.

        Returns:
            np.ndarray: Processed version of the input data.
        """
        return X

    def _fit_model(
        self,
        X_preprocessed: np.ndarray,
        **kwargs,
    ) -> None:
        self.tablevectorizer.fit(X_preprocessed)
        self._is_fitted = True

    def _compute_embeddings(
            self,
            X_train_preprocessed: np.ndarray,
            X_test_preprocessed: np.ndarray | None = None,
            outlier: bool = False,
            **kwargs
    ):
        """Compute embeddings using the fitted TableVectorizer.

        Args:
            X (np.ndarray): Input data to transform into embeddings.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: The computed embeddings.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if outlier:
            embeddings = self.tablevectorizer.transform(X_train_preprocessed)
            return embeddings.to_numpy(), None
        if self._is_fitted:

            X_train = pl.from_numpy(X_train_preprocessed)
            X_test = pl.from_numpy(X_test_preprocessed)

            embeddings_train = self.tablevectorizer.transform(X_train)
            embeddings_test = self.tablevectorizer.transform(X_test)

            return embeddings_train.to_numpy(), embeddings_test.to_numpy()
        else:
            raise ValueError("Model is not fitted.")

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Reinitializes the TableVectorizer and clears fitted state.
        """
        self.tablevectorizer = TableVectorizer()
        self._is_fitted = False
