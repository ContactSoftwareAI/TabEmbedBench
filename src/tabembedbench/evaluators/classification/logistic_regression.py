import numpy as np
from sklearn.linear_model import LogisticRegression

from tabembedbench.constants import CLASSIFICATION_TASKS, CLASSIFIER_OPTIMIZATION_METRIC
from tabembedbench.evaluators.abstractevaluator import AbstractHPOEvaluator


class LogisticRegressionHPOEvaluator(AbstractHPOEvaluator):
    """Evaluate Logistic Regression hyperparameters.

    This class serves as an evaluator for hyperparameter optimization of Logistic Regression
    models in the context of classification tasks. It defines the scoring metric, hyperparameter
    search space, and manages the generation of model predictions for downstream operations.

    Attributes:
        model_class (type): Indicates the model class being evaluated. Specifically, this is
            `LogisticRegression` from scikit-learn.
    """

    model_class = LogisticRegression

    def __init__(self, **kwargs) -> None:
        super().__init__(
            name="Logistic Regression",
            task_type=CLASSIFICATION_TASKS,
            **kwargs,
        )

    def get_scoring_metric(self) -> str:
        """Get the scoring metric for cross-validation.

        Returns the F1-weighted score, which balances precision and recall
        while accounting for class imbalance.

        Returns:
            str: Scoring metric string compatible with scikit-learn's cross_val_score.
        """
        return CLASSIFIER_OPTIMIZATION_METRIC

    def _get_search_space(self) -> dict[str, dict]:
        """Get the hyperparameter search space for Logistic Regression.

        Defines the optimization search space for:
        - C: Inverse regularization strength (1e-4 to 1e2, log scale)
        - penalty: Regularization type ('l2', 'l1')
        - solver: Optimization algorithm ('lbfgs', 'liblinear', 'saga')
        - max_iter: Maximum iterations (100 to 1000)
        - class_weight: Handling of class imbalance ('balanced', None)

        Returns:
            dict[str, dict]: Dictionary mapping hyperparameter names to their
                search space configurations.
        """
        return {
            "C": {
                "type": "float",
                "low": 1e-4,
                "high": 1e2,
                "log": True,
            },
            "penalty": {
                "type": "categorical",
                "choices": ["l2"],
            },
            "solver": {
                "type": "categorical",
                "choices": ["lbfgs"],
            },
            "max_iter": {
                "type": "int",
                "low": 100,
                "high": 1000,
                "step": 100,
            },
            "class_weight": {
                "type": "categorical",
                "choices": ["balanced", None],
            },
        }

    def _get_model_predictions(self, model, embeddings: np.ndarray) -> np.ndarray:
        """Get probability predictions from the trained model.

        Args:
            model: Trained LogisticRegression model instance.
            embeddings (np.ndarray): Input embeddings for prediction.

        Returns:
            np.ndarray: Probability predictions of shape (n_samples, n_classes).
        """
        return model.predict_proba(embeddings)
