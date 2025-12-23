import warnings
from typing import Optional

import numpy as np
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils._tags import Tags, TargetTags

from tabembedbench.constants import (
    CLASSIFICATION_TASKS,
    MAX_BEST_MODEL_ITERATIONS,
    MAX_HPO_ITERATIONS,
)
from tabembedbench.evaluators.abstractevaluator import AbstractHPOEvaluator
from tabembedbench.utils.torch_utils import get_device


class SklearnMLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for sklearn MLPClassifier with consistent interface.

    This wrapper ensures compatibility with the evaluation framework while
    using sklearn's native MLPClassifier implementation.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

        self.model = None

    def __sklearn_tags__(self):
        """Implement sklearn tags for compatibility with newer sklearn versions."""
        try:
            tags = Tags(
                estimator_type="classifier", target_tags=TargetTags(required=True)
            )
            return tags
        except (ImportError, TypeError):
            # Fallback for older sklearn versions or if Tags API changes
            return None

    def _more_tags(self):
        """Provide additional tags for sklearn compatibility."""
        return {"requires_y": True, "_estimator_type": "classifier"}

    @property
    def _estimator_type(self):
        """Return the estimator type."""
        return "classifier"

    def fit(self, X, y):
        """Fit the MLP classifier."""
        # Handle pandas DataFrames
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input X contains NaN or Inf values")
        if np.any(np.isnan(y)):
            raise ValueError("Input y contains NaN values")

        # Store classes BEFORE fitting
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.scaler_ = StandardScaler()
        # Scale features
        X_scaled = self.scaler_.fit_transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Create and fit the model
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state,
            )

            self.model.fit(X_scaled, y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        if hasattr(X, "values"):
            X = X.values

        X_scaled = self.scaler_.transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pred = self.model.predict(X_scaled)

        return pred

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        if hasattr(X, "values"):
            X = X.values

        X_scaled = self.scaler_.transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            proba = self.model.predict_proba(X_scaled)

        return proba


class MLPClassifierEvaluator(AbstractHPOEvaluator):
    """MLP Classifier evaluator with Optuna hyperparameter optimization using PyTorch.

    This evaluator uses a PyTorch-based Multi-Layer Perceptron for classification
    tasks with automatic hyperparameter optimization via Optuna.

    Attributes:
        device (str): Device to run the model on ('cpu' or 'cuda').
        input_dim (int | None): Number of input features (set during training).
        output_dim (int | None): Number of output classes (set during training).
    """

    model_class = SklearnMLPClassifierWrapper

    def __init__(
        self,
        n_trials: int = 10,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """Initialize the MLPClassifierEvaluator.

        Args:
            n_trials (int, optional): Number of optimization trials. Defaults to 50.
            cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.
        """
        super().__init__(
            name="MLPClassifier",
            task_type=CLASSIFICATION_TASKS,
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose,
        )
        self.input_dim = None
        self.output_dim = None

    def _get_search_space(self) -> dict:
        """Define the hyperparameter search space for sklearn MLP classifier."""
        return {
            "n_layers": {"type": "int", "low": 1, "high": 10},
            "hidden_layer_sizes": {
                "type": "int_sequence",
                "length_param": "n_layers",
                "low": 32,
                "high": 512,
                "log": True,
            },
            "alpha": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "learning_rate_init": {
                "type": "float",
                "low": 1e-6,
                "high": 1e-2,
                "log": True,
            },
            "batch_size": {
                "type": "categorical",
                "choices": [16, 32, 64, 128, 256, 512],
            },
            "max_iter": {
                "type": "constant",
                "value": MAX_HPO_ITERATIONS,
            },
            "activation": {
                "type": "categorical",
                "choices": ["relu", "tanh", "logistic"],
            },
        }

    def get_scoring_metric(self) -> str:
        """Return the scoring metric for classification."""
        return "neg_log_loss"

    def _get_model_predictions(self, model, embeddings: np.ndarray):
        """Get probability predictions from the model."""
        return model.predict_proba(embeddings)

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
        **kwargs,
    ) -> tuple:
        """Get predictions from the MLP classifier."""
        if train:
            if y is None:
                raise ValueError("y must be provided for training")

            # Set dimensions based on data
            self.input_dim = embeddings.shape[1]
            self.output_dim = len(np.unique(y))

        additional_parameters = {
            "max_iter": MAX_BEST_MODEL_ITERATIONS,
        }

        return super().get_prediction(embeddings, y, train, **additional_parameters)
