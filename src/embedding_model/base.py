from abc import ABC, abstractmethod
from typing import Union, Optional, Any, Dict, List

import numpy as np


class BaseEmbeddingGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Compute embeddings for the input data.

        Args:
            X: np.ndarray

        Returns:

        """
        pass
