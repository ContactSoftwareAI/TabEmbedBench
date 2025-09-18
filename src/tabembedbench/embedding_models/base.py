from abc import ABC, abstractmethod
import time

import numpy as np
import torch


class BaseEmbeddingGenerator(ABC):
    """Base class for generating embeddings using various models.

    This class provides an abstract base for embedding models for tabular data. It
    defines the necessary methods and properties that must be implemented by any
    specific embedding generator. It includes functionalities for preprocessing data,
    computing embeddings, managing the embedding model state, and handling naming
    conventions for specific instances of the embedding generator.

    Attributes:
        name (str): The name of the embedding generator instance that can be
                    retrieved or set.
    """

    def __init__(self):
        self._name = self._get_default_name()

    @property
    @abstractmethod
    def task_only(self) -> bool:
        """Property that defines whether the operation is restricted to task-only functionality.

        This abstract property must be implemented in subclasses to specify whether the operation
        should be exclusively task-oriented or involve broader functionalities.

        Returns:
            bool: True if the operation is restricted to task-only functionality,
                  False otherwise.
        """

    @abstractmethod
    def _get_default_name(self) -> str:
        """Defines an abstract method to be implemented by subclasses. This method should
        return a default name specific to the subclass implementation. It must be
        overridden by any concrete subclass.

        Raises:
            NotImplementedError: If not implemented by the subclass.

        Returns:
            str: A string representing the default name.
        """

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
    def _preprocess_data(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        """Preprocess the input data for training or inference.

        This method is designed to perform preprocessing operations on the
        input data. The preprocessing can vary depending on whether the
        input data is for training or inference. Specific preprocessing
        steps would depend on the concrete implementation of this method
        in a subclass.

        Args:
            X (np.ndarray): The input data to preprocess.
            train (bool, optional): Whether the data is for training. Default is True.

        Returns:
            np.ndarray: The processed data after applying preprocessing steps.
        """
        return X

    @abstractmethod
    def _compute_embeddings(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Computes the embeddings for the provided input data using a specified algorithm. This method must be
        implemented by subclasses to specify the process of generating embeddings. It takes a numerical array
        as input and returns a numerical array that represents the corresponding embeddings. It should be called on
        an array which has already been preprocessed by the preprocess_data method.

        Args:
            X (np.ndarray): Input data for which embeddings are to be computed.

        Returns:
            np.ndarray: Computed embeddings corresponding to the input data.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_embedding_model(self):
        """Defines an abstract method for resetting an embedding model. This method should be
        implemented by subclasses to provide functionality for resetting or reinitializing
        the state of the embedding model after a benchmark dataset has been processed.
        The implementation details should be specific to the embedding model being used.

        Args:
            None

        Raises:
            NotImplementedError: This method must be implemented by the subclass and will
            raise this error if accessed directly.
        """
        raise NotImplementedError

    def compute_embeddings(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray = None,
    ):
        """
        Computes embeddings for training and optionally test data. The embeddings are preprocessed
        and transformed, and the computation time required for embeddings is tracked.

        Args:
            X_train (np.ndarray): Data for training.
            X_test (np.ndarray, optional): Data for testing. Default is None.

        Returns:
            tuple: If `X_test` is provided, returns a tuple containing embeddings of
                training data, embeddings of test data, and the time taken to compute
                embeddings. Otherwise, returns a tuple containing embeddings of training
                data and the time taken to compute embeddings.
        """
        X_train = self._preprocess_data(X_train, train=True)

        start_time = time.time()
        X_train_embed = self._compute_embeddings(X_train)
        compute_embeddings_time = time.time() - start_time

        if self.check_emb_shape(X_train_embed):
            raise ValueError("The shape of the embeddings is not correct")

        if X_test is not None:
            X_test = self._preprocess_data(X_test, train=False)

            start_test_time = time.time()
            X_test_embed = self._compute_embeddings(X_test)
            compute_test_embeddings_time = time.time() - start_test_time

            return X_train_embed, X_test_embed, compute_embeddings_time, compute_test_embeddings_time
        else:
            return X_train_embed, compute_embeddings_time

    def check_emb_shape(self, X_train_embed):
        if len(X_train_embed.shape) != 2:
            return True
        else:
            return False


class Dummy(BaseEmbeddingGenerator):
    """Class representing a dummy embedding generator.

    This class is designed to showcase a basic embedding generator. It includes method
    stubs for preprocessing data and computing embeddings. It inherits from
    `BaseEmbeddingGenerator`.

    Attributes:
        name (str): The name of the embedding generator, set to a default value during initialization.
    """

    def __init__(self):
        self.name = self._get_default_name()

    @property
    def task_only(self) -> bool:
        return False

    def _get_default_name(self) -> str:
        return "Dummy"

    def preprocess_data(self, X: torch.Tensor | np.ndarray, **kwargs) -> np.ndarray:
        self.name = "DummyPreprocessed"
        return X

    def compute_embeddings(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        return np.random.rand(X.shape[0], 10)

    def reset_embedding_model(self):
        self.name = "DummyReset"
