from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingGenerator(ABC):
    def __init__(self):
        pass

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
