import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict

import numpy as np
import optuna
from sklearn.base import BaseEstimator


class AbstractEvaluator(ABC):
    def __init__(
            self,
            name: str,
            task_type: str
    ):
        self._name = name or self.__class__.__name__
        self.task_type = task_type

    @abstractmethod
    def get_prediction(
            self,
            embeddings: np.ndarray,
            y: np.ndarray | None = None,
            train: bool = True,
    ) -> tuple:
        pass

    @abstractmethod
    def reset_evaluator(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    def get_task(self):
        return self.task_type
