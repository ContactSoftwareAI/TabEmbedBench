from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseEmbeddingGenerator(ABC):
    def __init__(self):
        self._name = self._get_default_name()

    @property
    @abstractmethod
    def supports_train_test_splits(self) -> bool:
        """
        Indicates whether the implementation supports train-test splits, i.e., it fits and transforms the train data,
        but only transform the test data on the fitted process. This is an abstract property
        that must be implemented by subclasses.

        Returns:
            bool: True if train-test splits are supported, otherwise False.
        """
        pass

    @abstractmethod
    def _get_default_name(self) -> str:
        """
        Get the default name for this embedding generator.
        Subclasses must implement this method.

        Returns:
            str: The default name of the embedding generator.
        """
        pass

    @property
    def name(self) -> str:
        """
        Get the name of the embedding generator.

        Returns:
            str: The name of the embedding generator.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """
        Set the name of the embedding generator.

        Args:
            value: The new name for the embedding generator.
        """
        self._name = value

    @abstractmethod
    def preprocess_data(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        """
        Preprocesses the input data based on training or inference phase.

        This abstract method is intended to apply necessary preprocessing steps,
        such as scaling, normalization, or transformations, on the provided
        input dataset. The processing may differ depending on whether the data
        is being prepared for training or inference.

        Args:
            X (np.ndarray): Input data array to be preprocessed.
            train (bool): Indicates whether the preprocessing is for the
                training phase (`True`) or inference phase (`False`). Defaults
                to `True`.

        Returns:
            np.ndarray: Preprocessed data array.
        """
        return X

    @abstractmethod
    def compute_embeddings(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Compute embeddings for the input data.

        Args:
            X: np.ndarray

        Returns:
            np.ndarray: Embeddings for the input data.

        """
        raise NotImplementedError


class Dummy(BaseEmbeddingGenerator):
    """
    Class representing a dummy embedding generator.

    This class is designed to showcase a basic embedding generator. It includes method
    stubs for preprocessing data and computing embeddings. It inherits from
    `BaseEmbeddingGenerator`.

    Attributes:
        name (str): The name of the embedding generator, set to a default value during initialization.
    """

    def __init__(self):
        self.name = self._get_default_name()

    def _get_default_name(self) -> str:
        return "Dummy"

    def preprocess_data(
        self, X: Union[torch.Tensor, np.ndarray], **kwargs
    ) -> np.ndarray:
        return X

    def compute_embeddings(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return np.random.rand(X.shape[0], 10)
