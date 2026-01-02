import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier

from tabembedbench.evaluators import AbstractEvaluator, AbstractHPOEvaluator


class KNNClassifierEvaluator(AbstractEvaluator):
    """K-Nearest Neighbors classifier evaluator for embedding quality assessment.

    This evaluator uses scikit-learn's KNeighborsClassifier to evaluate the quality
    of embeddings on classification tasks. It trains a KNN classifier on the embeddings
    and returns probability predictions.

    Attributes:
        num_neighbors (int): Number of nearest neighbors to use for classification.
        model_params (dict): Dictionary containing all KNN model parameters including
            weights and metric.
        knn_classifier (KNeighborsClassifier): The underlying scikit-learn KNN classifier.
    """

    def __init__(
        self,
        num_neighbors: int,
        weights: str,
        metric: str,
        other_model_params: dict = {},
    ):
        """Initialize the KNN classifier evaluator.

        Args:
            num_neighbors (int): Number of nearest neighbors to consider.
            weights (str): Weight function used in prediction. Possible values:
                'uniform': All points in each neighborhood are weighted equally.
                'distance': Weight points by the inverse of their distance.
            metric (str): Distance metric to use for the tree. Common options include
                'euclidean', 'manhattan', 'cosine', etc.
            other_model_params (dict, optional): Additional parameters to pass to
                KNeighborsClassifier. Defaults to {}.
        """
        super().__init__(
            name="KNNClassifier",
            task_type=[
                "Supervised Binary Classification",
                "Supervised Multiclass Classification",
            ],
        )
        self.num_neighbors = num_neighbors

        self.model_params = dict(other_model_params.items())

        self.model_params["weights"] = weights
        self.model_params["metric"] = metric

        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=self.num_neighbors, **self.model_params
        )

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
    ):
        """Get probability predictions from the KNN classifier.

        Args:
            embeddings (np.ndarray): Input embeddings of shape (n_samples, n_features).
            y (np.ndarray | None, optional): Target labels for training. Required if
                train=True. Defaults to None.
            train (bool, optional): Whether to train the classifier before prediction.
                Defaults to True.

        Returns:
            tuple: A tuple containing:
                - predictions (np.ndarray): Probability predictions of shape
                    (n_samples, n_classes).
                - additional_info (None): No additional information is returned.

        Raises:
            ValueError: If train=True and y is None.
        """
        if train:
            if y is None:
                raise ValueError("y must be provided for training")
            self.knn_classifier.fit(embeddings, y)

            return self.knn_classifier.predict_proba(embeddings), None
        return self.knn_classifier.predict_proba(embeddings), None

    def reset_evaluator(self):
        """Reset the evaluator to its initial state.

        Reinitializes the KNN classifier with the original parameters, clearing
        any trained model state.
        """
        self.knn_classifier = KNeighborsClassifier(
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


class KNNClassifierEvaluatorHPO(AbstractHPOEvaluator):
    def get_scoring_metric(self) -> dict[str, str]:
        """Return the scoring metric for classification."""
        return "f1_weighted"

    def _get_search_space(self) -> dict[str, optuna.search_space]:
        return {
            "n_neighbors": {"type": "int", "low": 5, "high": 100, "step": 5},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "metric": {
                "type": "categorical",
                "choices": ["euclidean", "manhattan", "cosine"],
            },
        }

    def _get_model_predictions(self, model, embeddings: np.ndarray):
        return model.predict_proba(embeddings)
