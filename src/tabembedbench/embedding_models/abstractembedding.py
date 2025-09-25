import time
from abc import ABC, abstractmethod

import numpy as np


class AbstractEmbeddingGenerator(ABC):
    """Abstract base class for generating embeddings from input data.

    This abstract class defines the structure required for subclasses to generate
    embeddings from input data. It provides methods for preprocessing, validating,
    and computing embeddings, which must be implemented in any concrete subclass.
    Additionally, the class includes methods for checking embedding data validity
    and resetting the embedding model. This serves as a framework for various
    embedding generation techniques and ensures consistency in their implementation.

    Attributes:
        name (str): The name associated with the instance.
    """
    def __init__(
            self,
            name: str,
    ):
        self._name = name

    @property
    @abstractmethod
    def task_only(self) -> bool:
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Gets the name attribute value.

        This property retrieves the value of the `_name` attribute,
        which represents the name associated with the instance.

        Returns:
            str: The name associated with the instance.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Sets the name property.

        Args:
            value (str): The value to assign to the name property.
        """
        self._name = value

    @abstractmethod
    def _preprocess_data(
            self,
            X: np.ndarray,
            train: bool = True,
            outlier: bool = False,
    ) -> np.ndarray:
        """Preprocess the input data for training or inference.

        This method is designed to perform preprocessing operations on the
        input data. The preprocessing can vary depending on whether the
        input data is for training or inference. Specific preprocessing
        steps would depend on the concrete implementation of this method
        in a subclass.

        Args:
            X (np.ndarray): The input data to preprocess.
            train (bool, optional): Whether the data is for training.
                Default is True.

        Returns:
            np.ndarray: The processed data after applying preprocessing steps.
        """
        return X

    @abstractmethod
    def _compute_embeddings(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Computes embeddings for the given input data.

        This is an abstract method that must be implemented by subclasses. The method
        is designed to process the input data, represented as a NumPy array, and
        return a NumPy array containing computed embeddings. The actual computation
        logic is determined by the specific implementation in the subclass.

        Args:
            X (np.ndarray): A NumPy array representing the input data for which
                embeddings need to be computed. The input dimensions and structure
                are dependent on the specific implementation.

        Returns:
            np.ndarray: A NumPy array representing the computed embeddings. The
                dimensions and structure of the returned array are determined by
                the implementation in the subclass.

        Raises:
            NotImplementedError: Raised when this abstract method is called directly
                without being overridden in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_embedding_model(self):
        """
        Abstract method to reset the embedding model.

        This method needs to be implemented by any subclass inheriting this base class.
        The purpose of this method is to reset or reinitialize the embedding model
        to ensure correct functioning or state restoration within the subclass.

        Raises:
            NotImplementedError: This exception is raised if the method is not implemented
                in a subclass.
        """
        raise NotImplementedError

    @staticmethod
    def _check_emb_shape(X_embed: np.ndarray) -> bool:
        """Verifies the shape of the embedding array to ensure it meets the expected
        requirements.

        Args:
            X_embed (np.ndarray): Input embedding array to be checked. Expected to
                be an array with shape (num_samples, embedding_dim).

        Returns:
            bool: Returns True if the shape of the array does not meet the
                requirement of being 2-dimensional, otherwise False.
        """
        return len(X_embed.shape) != 2

    @staticmethod
    def _check_nan(X_embed: np.ndarray) -> bool:
        """Checks if the given array contains any NaN (Not a Number) values.

        This static method evaluates the given NumPy array for the presence of any
        NaN values. It is used internally to ensure that the data array provided
        to calculations or operations is valid and does not contain invalid entries.

        Args:
            X_embed (np.ndarray): The NumPy array that needs to be checked for NaN
                values.

        Returns:
            bool: True if the array contains any NaN values; otherwise, False.
        """
        return np.isnan(X_embed).any()

    @staticmethod
    def _validate_embeddings(
            X_train_embed: np.ndarray,
            X_test_embed: np.ndarray | None = None
    ) -> bool:
        """Validates the embeddings for training and test datasets.

        This method ensures that the embeddings for both training and test datasets
        have the correct shape and do not contain NaN values.

        Args:
            X_train_embed (np.ndarray): The embedding matrix for the training dataset.
            X_test_embed (Optional[np.ndarray]): The embedding matrix for the test
                dataset. Defaults to None.

        Returns:
            bool: True if the embeddings for both datasets are valid, otherwise False.
        """
        train_valid = (
                AbstractEmbeddingGenerator._check_emb_shape(X_train_embed) and
                AbstractEmbeddingGenerator._check_nan(X_train_embed)
        )
        if X_test_embed is not None:
            test_valid = (
                    AbstractEmbeddingGenerator._check_emb_shape(X_test_embed) and
                    AbstractEmbeddingGenerator._check_nan(X_test_embed)
            )
            return train_valid and test_valid
        return train_valid

    def compute_embeddings(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray = None,
        outlier: bool = False,
    ) -> tuple:
        """
        Computes embeddings for the given training and optional testing data.

        The method processes the input data using a preprocessing step before generating
        embeddings. If testing data is provided, it computes embeddings for both training
        and testing data and validates them. Execution times for the embedding computation
        steps are also returned.

        Args:
            X_train (np.ndarray): The training data to compute embeddings for.
            X_test (np.ndarray, optional): The testing data to compute embeddings for.
                Default is None.
            outlier (bool): A flag to indicate if outlier handling is required
                during preprocessing. Default is False.

        Returns:
            tuple: A tuple containing the following:
                - X_train_embed (np.ndarray): The embeddings for the training data.
                - X_test_embed (np.ndarray, optional): The embeddings for the testing
                  data if provided.
                - compute_embeddings_time (float): Time taken to compute embeddings
                  for the training data.
                - compute_test_embeddings_time (float, optional): Time taken to compute
                  embeddings for the testing data if provided.

        Raises:
            Exception: If the embeddings generated for training and testing data
                contain NaN values.
        """
        X_train = self._preprocess_data(X_train, train=True, outlier=outlier)

        start_time = time.time()
        X_train_embed = self._compute_embeddings(X_train)
        compute_embeddings_time = time.time() - start_time

        if X_test is not None:
            X_test = self._preprocess_data(X_test, train=False)

            start_test_time = time.time()
            X_test_embed = self._compute_embeddings(X_test)
            compute_test_embeddings_time = time.time() - start_test_time

            if not self._validate_embeddings(X_train_embed, X_test_embed):
                raise Exception("Embeddings contain NaN values.")

            return (X_train_embed, X_test_embed, compute_embeddings_time,
                    compute_test_embeddings_time)
        return X_train_embed, compute_embeddings_time
