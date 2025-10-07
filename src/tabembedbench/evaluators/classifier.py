import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from tabembedbench.evaluators import AbstractEvaluator


class KNNClassifierEvaluator(AbstractEvaluator):
    def __init__(
            self,
            num_neighbors: int,
            weights: str,
            metric: str,
            other_model_params: dict = {},
    ):
        super().__init__(
            name="KNNClassifier",
            task_type="Supervised Classification"
        )
        self.num_neighbors = num_neighbors

        self.model_params = dict(other_model_params.items())

        self.model_params["weights"] = weights
        self.model_params["metric"] = metric

        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=self.num_neighbors,
            **self.model_params
        )

    def get_prediction(
            self,
            embeddings: np.ndarray,
            y: np.ndarray | None = None,
            train: bool = True,
    ):
        if train:
            if y is None:
                raise ValueError("y must be provided for training")
            self.knn_classifier.fit(embeddings, y)

            return self.knn_classifier.predict_proba(embeddings), None
        return self.knn_classifier.predict_proba(embeddings), None

    def reset_evaluator(self):
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=self.num_neighbors,
            **self.model_params
        )

    def get_parameters(self):
        params = dict(self.model_params.items())

        params["num_neighbors"] = self.num_neighbors

        return params
