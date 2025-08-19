from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingGenerator(ABC):
    def __init__(self):
        self._name = self._get_default_name()

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
