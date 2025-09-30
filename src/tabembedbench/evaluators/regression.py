import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from tabembedbench.evaluators import AbstractEvaluator

class KNNRegressorEvaluator(AbstractEvaluator):
    def __init__(
            self,
            num_neighbors: int,
            weights: str,
            metric: str,
            other_model_params: dict = {},
    ):
        super().__init__(
            name="KNNRegressor",
            task_type="Supervised Regression"
        )
        self.num_neighbors = num_neighbors

        self.model_params = dict(other_model_params.items())

        self.model_params["weights"] = weights
        self.model_params["metric"] = metric

        self.knn_regressor = KNeighborsRegressor(
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
            self.knn_regressor.fit(embeddings, y)

            return self.knn_regressor.predict(embeddings), None
        else:
            return self.knn_regressor.predict(embeddings), None

    def reset_evaluator(self):
        self.knn_regressor = KNeighborsRegressor(
            n_neighbors=self.num_neighbors,
            **self.model_params
        )

    def get_parameters(self):
        params = dict(self.model_params.items())

        params["num_neighbors"] = self.num_neighbors

        return params
