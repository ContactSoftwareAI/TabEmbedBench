import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from tabembedbench.evaluators import AbstractEvaluator

class LocalOutlierFactorEvaluator(AbstractEvaluator):
    def __init__(
            self,
            model_params: dict,
    ):
        super().__init__(
            name="LocalOutlierFactor",
            task_type="Outlier Detection"
        )

        self.model_params = model_params

        self.lof = LocalOutlierFactor(
            **self.model_params
        )

    def get_prediction(
            self,
            embeddings: np.ndarray,
            y = None,
            train = True,
    ):
        self.lof.fit(embeddings)
        prediction = (-1)*self.lof.negative_outlier_factor_

        return prediction, None

    def reset_evaluator(self):
        self.lof = LocalOutlierFactor(
            **self.model_params
        )

    def get_parameters(self):
        return self.model_params


class IsolationForestEvaluator(AbstractEvaluator):
    def __init__(
            self,
            model_params: dict,
            random_seed: int = 42
    ):
        super().__init__(
            name="IsolationForest",
            task_type="Outlier Detection")

        self.model_params = model_params
        self.random_seed = random_seed

        self.iso_forest = IsolationForest(
            **self.model_params,
            random_state=self.random_seed
        )

    def get_prediction(
            self,
            embeddings: np.ndarray,
            y = None,
            train = True,
    ):
        self.iso_forest.fit(embeddings)
        prediction = (-1)*self.iso_forest.decision_function(embeddings)

        return prediction, None

    def reset_evaluator(self):
        self.iso_forest = IsolationForest(
            **self.model_params,
            random_state=self.random_seed
        )

    def get_parameters(self):
        return self.model_params



