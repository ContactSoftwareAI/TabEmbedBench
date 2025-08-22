from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class BaseEmbeddingGenerator(ABC):
    """
    Base class for generating embeddings using various models.

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
        pass

    @abstractmethod
    def _get_default_name(self) -> str:
        pass

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @abstractmethod
    def preprocess_data(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        return X

    @abstractmethod
    def compute_embeddings(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reset_embedding_model(self):
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

    @property
    def task_only(self) -> bool:
        return False

    def _get_default_name(self) -> str:
        return "Dummy"

    def preprocess_data(
        self, X: Union[torch.Tensor, np.ndarray], **kwargs
    ) -> np.ndarray:
        return X

    def compute_embeddings(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return np.random.rand(X.shape[0], 10)

    def reset_embedding_model(self):
        pass
