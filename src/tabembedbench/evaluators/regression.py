import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from tabembedbench.evaluators import AbstractEvaluator


class KNNRegressorEvaluator(AbstractEvaluator):
    """K-Nearest Neighbors regressor evaluator for embedding quality assessment.

    This evaluator uses scikit-learn's KNeighborsRegressor to evaluate the quality
    of embeddings on regression tasks. It trains a KNN regressor on the embeddings
    and returns continuous value predictions.

    Attributes:
        num_neighbors (int): Number of nearest neighbors to use for regression.
        model_params (dict): Dictionary containing all KNN model parameters including
            weights and metric.
        knn_regressor (KNeighborsRegressor): The underlying scikit-learn KNN regressor.
    """

    def __init__(
        self,
        num_neighbors: int,
        weights: str,
        metric: str,
        other_model_params: dict = {},
    ):
        """Initialize the KNN regressor evaluator.

        Args:
            num_neighbors (int): Number of nearest neighbors to consider.
            weights (str): Weight function used in prediction. Possible values:
                'uniform': All points in each neighborhood are weighted equally.
                'distance': Weight points by the inverse of their distance.
            metric (str): Distance metric to use for the tree. Common options include
                'euclidean', 'manhattan', 'cosine', etc.
            other_model_params (dict, optional): Additional parameters to pass to
                KNeighborsRegressor. Defaults to {}.
        """
        super().__init__(name="KNNRegressor", task_type="Supervised Regression")
        self.num_neighbors = num_neighbors

        self.model_params = dict(other_model_params.items())

        self.model_params["weights"] = weights
        self.model_params["metric"] = metric

        self.knn_regressor = KNeighborsRegressor(
            n_neighbors=self.num_neighbors, **self.model_params
        )

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
    ):
        """Get continuous value predictions from the KNN regressor.

        Args:
            embeddings (np.ndarray): Input embeddings of shape (n_samples, n_features).
            y (np.ndarray | None, optional): Target values for training. Required if
                train=True. Defaults to None.
            train (bool, optional): Whether to train the regressor before prediction.
                Defaults to True.

        Returns:
            tuple: A tuple containing:
                - predictions (np.ndarray): Continuous value predictions of shape
                    (n_samples,).
                - additional_info (None): No additional information is returned.

        Raises:
            ValueError: If train=True and y is None.
        """
        if train:
            self.knn_regressor.fit(embeddings, y)

            return self.knn_regressor.predict(embeddings), None
        else:
            return self.knn_regressor.predict(embeddings), None

    def reset_evaluator(self):
        """Reset the evaluator to its initial state.

        Reinitializes the KNN regressor with the original parameters, clearing
        any trained model state.
        """
        self.knn_regressor = KNeighborsRegressor(
            n_neighbors=self.num_neighbors, **self.model_params
        )

    def get_parameters(self):
        """Get the current parameters of the evaluator.

        Returns:
            dict: Dictionary containing all evaluator parameters including
                num_neighbors, weights, metric, and any additional model parameters.
        """
        params = dict(self.model_params.items())

        params["num_neighbors"] = self.num_neighbors

        return params
