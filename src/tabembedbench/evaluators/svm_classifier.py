import numpy as np
from sklearn.svm import SVC

from tabembedbench.constants import CLASSIFIER_OPTIMIZATION_METRIC
from tabembedbench.evaluators.abstractevaluator import AbstractHPOEvaluator


class SupportVectorClassification(AbstractHPOEvaluator):
    model_class = SVC

    def get_scoring_metric(self) -> str:
        return CLASSIFIER_OPTIMIZATION_METRIC

    def _get_search_space(self) -> dict[str, dict]:
        return {
            "C": {"type": "float", "low": 1e-3, "high": 10, "log": True},
            "kernel": {"type": "constant", "value": "rbf"},
            "gamma": {"type": "float", "low": 1e-3, "high": 10, "log": True},
            "degree": {"type": "int", "low": 2, "high": 5},
        }

    def _get_model_predictions(self, model, embeddings: np.ndarray) -> np.ndarray:
        return model.predict_proba(embeddings)
