import logging
import time
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl


class AbstractEmbeddingGenerator(ABC):
    """
    An abstract base class for embedding generators.

    This class defines a blueprint for embedding generation logic for data preprocessing,
    model training, and embedding computation. It includes abstract methods that subclasses
    must implement to specify their behavior and functionality. Utility methods are also
    provided to help with embedding validation and handling.

    Attributes:
        name (str): The name assigned to the embedding generator instance.
        is_self_contained (bool): Indicates whether the generator is self-contained,
            managing its own resources independently.
    """

    def __init__(
        self,
        name: str,
        is_end_to_end_model: bool = False,
        end_to_end_compatible_tasks: list[str] | None = None,
        max_num_samples: int = 100000,
        max_num_features: int = 500,
    ):
        """
        Initializes the object with the provided name and a flag indicating whether it is
        self-contained.

        Args:
            name (str): The name assigned to the object.
            is_self_contained (bool): A flag indicating whether the object is self-contained.
                Defaults to False.
            compatible_tasks_for_end_to_end (list[str] | None): A list of task types supported by the model.
        """
        self._name = name
        self._is_fitted = False
        self._is_end_to_end_model = is_end_to_end_model
        self._logger = logging.getLogger(__name__)
        self._end_to_end_compatible_tasks = end_to_end_compatible_tasks
        self.max_num_samples = max_num_samples
        self.max_num_features = max_num_features

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
    def name(self, new_name: str):
        self._name = new_name

    @property
    def is_end_to_end_model(self) -> bool:
        """Gets whether this model is self-contained.

        Returns:
            bool: True if the model solves tasks directly without evaluators, False otherwise.
        """
        return self._is_end_to_end_model

    @property
    def end_to_end_compatible_tasks(self):
        return self._end_to_end_compatible_tasks

    @end_to_end_compatible_tasks.setter
    def end_to_end_compatible_tasks(self, tasks: list[str]):
        self._end_to_end_compatible_tasks = tasks

    # ========== Abstract Methods (Subclasses must implement) ==========
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

    def _get_prediction(self, X: np.ndarray, **kwargs) -> np.ndarray:
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

    def check_dataset_constraints(self, num_samples: int, num_features: int):
        return (
            num_samples <= self.max_num_samples
            and num_features <= self.max_num_features
        )

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

    def preprocess_data(
        self,
        X_train: np.ndarray | pl.DataFrame | pd.DataFrame,
        X_test: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs,
    ):
        """
        Preprocesses the training and optional test datasets, applies transformations, and fits
        a model. Handles optional outlier handling and additional preprocessing options.

        Args:
            X_train (np.ndarray | pl.DataFrame | pd.DataFrame): The training dataset to be
                preprocessed.
            X_test (np.ndarray | None): The optional test dataset to be preprocessed. Defaults
                to None.
            y_train:
            outlier (bool): Flag to enable or disable outlier processing. Defaults to False.
            **kwargs: Additional keyword arguments for preprocessing or model configuration.

        Returns:
            Tuple[np.ndarray | pl.DataFrame | pd.DataFrame, np.ndarray | pl.DataFrame | pd.DataFrame | None]:
                A tuple containing the preprocessed training and (if provided) test datasets.
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

        self._fit_model(
            X_train_preprocessed, y_preprocessed=y_train, outlier=outlier, **kwargs
        )

        return X_train_preprocessed, X_test_preprocessed

    def generate_embeddings(
        self,
        X_train: np.ndarray | pl.DataFrame | pd.DataFrame,
        X_test: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray | None, float]:
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
                  for the training and test data.

        Raises:
            Exception: If the embeddings generated for training and testing data
                contain NaN values.
        """
        X_train_preprocessed, X_test_preprocessed = self.preprocess_data(
            X_train, X_test, outlier=outlier, **kwargs
        )
        start_time = time.perf_counter()
        train_embeddings, test_embeddings = self._compute_embeddings(
            X_train_preprocessed, X_test_preprocessed, outlier=outlier, **kwargs
        )
        compute_embeddings_time = time.perf_counter() - start_time

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

    def get_end_to_end_prediction(
        self,
        X_train: np.ndarray | pl.DataFrame | pd.DataFrame,
        y_train: np.ndarray,
        X_test: np.ndarray | None = None,
        task_type: str = "Supervised Classification",
        **kwargs,
    ) -> np.ndarray:
        if not self._is_end_to_end_model:
            raise NotImplementedError(
                "get_end_to_end_prediction() is only available for end-to-end models."
            )
        if (
            self.end_to_end_compatible_tasks
            and task_type not in self.end_to_end_compatible_tasks
        ):
            raise ValueError()

        _, X_test_preprocessed = self.preprocess_data(
            X_train, X_test=X_test, y_train=y_train, task_type=task_type, **kwargs
        )
        return self._get_prediction(X_test_preprocessed, task_type=task_type)
