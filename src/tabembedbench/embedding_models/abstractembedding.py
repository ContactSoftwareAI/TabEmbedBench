import time
import logging
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
        """Initializes an instance of the AbstractEmbeddingGenerator class.

        Args:
            name (str): The name associated with the instance.
        """
        self._name = name
        self._is_fitted = False
        self._logger = logging.getLogger(__name__)

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
        **kwargs,
    ) -> np.ndarray:
        """Preprocesses the input data.

        This method must be implemented in derived classes. The method is
        designed to prepare the data before the `_compute_embeddings` method.
        The actual preprocess logic is determined by the specific implementation
        in the subclass.

        Args:
            X (np.ndarray): The input data to preprocess.
            train (bool): A flag indicating whether the preprocessing is for training.
                Default is True.
            outlier (bool): A flag indicating whether to handle outliers during
                preprocessing. Default is False.

        Returns:
            np.ndarray: The preprocessed data.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _fit_model(
        self,
        X_preprocessed: np.ndarray,
        y_preprocessed: np.ndarray | None = None,
        train: bool = True,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray,
        X_test_preprocessed: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute embeddings for the provided preprocessed data.

        This is an abstract method that must be implemented in a subclass. It is responsible
        for generating the embeddings for the training data and optionally for the test data.

        Args:
            X_train_preprocessed (np.ndarray): Preprocessed training data used to compute
                embeddings.
            X_test_preprocessed (np.ndarray | None): Preprocessed test data used to compute
                embeddings. If None, embeddings are generated only for the training data.
            **kwargs: Additional keyword arguments for embedding computation.

        Returns:
            np.ndarray: Computed embeddings for the provided data.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_embedding_model(self, *args, **kwargs):
        """Abstract method to reset the embedding model.

        This method needs to be implemented by any subclass inheriting this base class.
        The purpose of this method is to reset or reinitialize the embedding model
        to ensure correct functioning or state restoration within the subclass.

        Raises:
            NotImplementedError: This exception is raised if the method is not
                implemented in a subclass.
        """
        raise NotImplementedError

    @staticmethod
    def _check_emb_shape(embeddings: np.ndarray) -> bool:
        """Checks whether the provided embedded data has the correct shape.

        The method validates if the input array `X_embed` has two dimensions as
        expected for embedding matrices. If the array does not have two dimensions,
        the shape is considered invalid.

        Args:
            embeddings (np.ndarray): The embedding matrix to validate.

        Returns:
            bool: Returns True if the shape is invalid (not 2-dimensional),
                  otherwise False.
        """
        return len(embeddings.shape) == 2

    @staticmethod
    def _check_nan(embeddings: np.ndarray) -> bool:
        """Checks if the given array contains any NaN (Not a Number) values.

        This static method evaluates the given NumPy array for the presence of any
        NaN values. It is used internally to ensure that the data array provided
        to calculations or operations is valid and does not contain invalid entries.

        Args:
            embeddings (np.ndarray): The NumPy array that needs to be checked
            for NaN
                values.

        Returns:
            bool: True if the array contains any NaN values; otherwise, False.
        """
        return not np.isnan(embeddings).any()

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
    ) -> bool:
        """Validates the embeddings for training and test datasets.

        This method ensures that the embeddings for both training and test datasets
        have the correct shape and do not contain NaN values.

        Args:
            embeddings (np.ndarray): The embedding matrix for the training
                dataset.

        Returns:
            bool: True if the embeddings for both datasets are valid, otherwise False.
        """
        shape_check = AbstractEmbeddingGenerator._check_emb_shape(embeddings)
        nan_check = AbstractEmbeddingGenerator._check_nan(embeddings)

        return shape_check and nan_check

    def generate_embeddings(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs,
    ):
        """Generates embeddings for the given training and optional testing
        data.

        The method processes the input data using a preprocessing step before
        generating embeddings. If testing data is provided, it computes
        embeddings for both training and testing data and validates them.
        Execution times for the embedding computation steps are also returned.

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
        X_train_preprocessed = self._preprocess_data(
            X_train, train=True, outlier=outlier, **kwargs
        )

        if X_test is not None:
            X_test_preprocessed = self._preprocess_data(
                X_test, train=False, outlier=outlier, **kwargs
            )
        else:
            X_test_preprocessed = None

        self._fit_model(X_train_preprocessed, outlier=outlier, **kwargs)

        start_time = time.time()
        train_embeddings, test_embeddings = self._compute_embeddings(
            X_train_preprocessed,
            X_test_preprocessed,
            outlier=outlier,
            **kwargs
        )
        compute_embeddings_time = time.time() - start_time

        if not self._validate_embeddings(train_embeddings):
            raise ValueError("Train Embeddings contain NaN values.")
        if test_embeddings is not None:
            if not self._validate_embeddings(test_embeddings):
                raise ValueError("Test Embeddings contain NaN values.")

        self._reset_embedding_model()

        return (
            train_embeddings,
            test_embeddings,
            compute_embeddings_time,
        )
