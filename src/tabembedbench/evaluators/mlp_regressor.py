from typing import Optional
import warnings
import numpy as np
import optuna
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._tags import Tags, TargetTags
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from tabembedbench.evaluators.abstractevaluator import AbstractHPOEvaluator
from tabembedbench.utils.torch_utils import get_device


class SklearnMLPRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for sklearn MLPRegressor with consistent interface."""

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

        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted_ = False

    def __sklearn_tags__(self):
        """Implement sklearn tags for compatibility with newer sklearn versions."""
        try:
            tags = Tags(
                estimator_type="regressor", target_tags=TargetTags(required=True)
            )
            return tags
        except (ImportError, TypeError):
            # Fallback for older sklearn versions or if Tags API changes
            return None

    @property
    def _estimator_type(self):
        """Return the estimator type."""
        return "regressor"

    def fit(self, X, y):
        """Fit the MLP regressor."""
        # Handle pandas DataFrames
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Create and fit the model
            self.model = MLPRegressor(
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

        X_scaled = self.scaler.transform(X)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pred = self.model.predict(X_scaled)

        return pred


class MLPRegressorEvaluator(AbstractHPOEvaluator):
    """MLP Regressor evaluator with Optuna hyperparameter optimization using sklearn."""

    def __init__(
        self,
        n_trials: int = 10,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__(
            name="MLPRegressor",
            task_type="Supervised Regression",
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose,
        )
        self.input_dim = None

    def _get_search_space(self) -> dict:
        """Define the hyperparameter search space for sklearn MLP regressor."""
        return {
            "n_layers": {"type": "int", "low": 1, "high": 3},
            "hidden_dim_base": {"type": "int", "low": 32, "high": 512, "log": True},
            "alpha": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "learning_rate_init": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True,
            },
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128, 256]},
            "max_iter": {"type": "int", "low": 50, "high": 200},
            "activation": {
                "type": "categorical",
                "choices": ["relu", "tanh", "identity"],
            },
        }

    def create_model(self, trial: optuna.Trial):
        """Create sklearn MLP model with hyperparameters suggested by Optuna."""
        search_space = self._get_search_space()

        # Number of hidden layers
        n_layers = trial.suggest_int(
            "n_layers",
            search_space["n_layers"]["low"],
            search_space["n_layers"]["high"],
        )

        # Hidden layer dimensions (tuple for sklearn)
        hidden_layer_sizes = tuple(
            trial.suggest_int(
                f"hidden_dim_{i}",
                search_space["hidden_dim_base"]["low"],
                search_space["hidden_dim_base"]["high"],
                log=search_space["hidden_dim_base"]["log"],
            )
            for i in range(n_layers)
        )

        # Alpha (L2 regularization)
        alpha = trial.suggest_float(
            "alpha",
            search_space["alpha"]["low"],
            search_space["alpha"]["high"],
            log=search_space["alpha"]["log"],
        )

        # Learning rate
        learning_rate_init = trial.suggest_float(
            "learning_rate_init",
            search_space["learning_rate_init"]["low"],
            search_space["learning_rate_init"]["high"],
            log=search_space["learning_rate_init"]["log"],
        )

        # Batch size
        batch_size = trial.suggest_categorical(
            "batch_size", search_space["batch_size"]["choices"]
        )

        # Max iterations
        max_iter = trial.suggest_int(
            "max_iter",
            search_space["max_iter"]["low"],
            search_space["max_iter"]["high"],
        )

        # Activation function
        activation = trial.suggest_categorical(
            "activation", search_space["activation"]["choices"]
        )

        return SklearnMLPRegressorWrapper(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=self.random_state,
        )

    def get_scoring_metric(self) -> str:
        """Return the scoring metric for regression."""
        return "neg_mean_squared_error"

    def _get_model_predictions(self, model, embeddings: np.ndarray):
        """Get predictions from the model."""
        return model.predict(embeddings)

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
    ) -> tuple:
        """Get predictions from the MLP regressor."""
        if train:
            if y is None:
                raise ValueError("y must be provided for training")

            # Set input dimension based on data
            self.input_dim = embeddings.shape[1]

        return super().get_prediction(embeddings, y, train)
