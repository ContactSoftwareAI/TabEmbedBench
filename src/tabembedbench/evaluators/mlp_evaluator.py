from typing import Optional

import numpy as np
import optuna
from optuna import trial
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler

from tabembedbench.evaluators.abstractevaluator import AbstractHPOEvaluator
from tabembedbench.utils.torch_utils import get_device


class PyTorchMLPWrapper(BaseEstimator):
    """Scikit-learn compatible wrapper for PyTorch MLP.

    This wrapper allows PyTorch MLP models to work with scikit-learn's
    cross_val_score and other sklearn utilities.

    Attributes:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): List of hidden layer dimensions.
        output_dim (int): Number of output units.
        dropout (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of training epochs.
        task_type (str): Type of task ('classification' or 'regression').
        device (str): Device to run the model on ('cpu' or 'cuda').
        early_stopping_patience (int): Number of epochs to wait before early stopping.
        model (nn.Sequential | None): The PyTorch model.
        scaler (StandardScaler): Feature scaler.
        is_fitted_ (bool): Whether the model has been fitted.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        task_type: str = "classification",
        device: Optional[str] = None,
        early_stopping_patience: int = 10,
    ):
        """Initialize the PyTorchMLPWrapper.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list[int]): List of hidden layer dimensions.
            output_dim (int): Number of output units.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.0.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            task_type (str, optional): Type of task ('classification' or 'regression').
                Defaults to "classification".
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                Defaults to "cpu".
            early_stopping_patience (int, optional): Number of epochs to wait before
                early stopping. Defaults to 10.
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.task_type = task_type
        self.device = device if device is not None else get_device()
        self.early_stopping_patience = early_stopping_patience

        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted_ = False

    def _build_model(self):
        """Build the PyTorch MLP model.

        Returns:
            nn.Sequential: The constructed MLP model.
        """
        layers = []
        prev_dim = self.input_dim

        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))

        # Add activation for classification
        if self.task_type == "classification" and self.output_dim > 1:
            layers.append(nn.Softmax(dim=1))

        return nn.Sequential(*layers)

    def fit(self, X, y):
        """Fit the MLP model.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Training targets of shape (n_samples,).

        Returns:
            PyTorchMLPWrapper: The fitted model instance.
        """
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        if self.task_type == "classification":
            y_tensor = torch.LongTensor(y).to(self.device)
            criterion = nn.CrossEntropyLoss()
        else:
            y_tensor = torch.FloatTensor(y).to(self.device)
            criterion = nn.MSELoss()

        # Build model
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                if self.task_type == "classification":
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(1), batch_y)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Make predictions.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predictions. For classification, returns class labels.
                For regression, returns predicted values.

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            if self.task_type == "classification":
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            else:
                predictions = outputs.squeeze(1).cpu().numpy()

        return predictions

    def predict_proba(self, X):
        """Predict class probabilities (for classification only).

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).

        Raises:
            ValueError: If task_type is not 'classification' or model is not fitted.
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")

        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probabilities = self.model(X_tensor).cpu().numpy()

        return probabilities


class MLPClassifierEvaluator(AbstractHPOEvaluator):
    """MLP Classifier evaluator with Optuna hyperparameter optimization using PyTorch.

    This evaluator uses a PyTorch-based Multi-Layer Perceptron for classification
    tasks with automatic hyperparameter optimization via Optuna.

    Attributes:
        device (str): Device to run the model on ('cpu' or 'cuda').
        input_dim (int | None): Number of input features (set during training).
        output_dim (int | None): Number of output classes (set during training).
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize the MLPClassifierEvaluator.

        Args:
            n_trials (int, optional): Number of optimization trials. Defaults to 50.
            cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                Defaults to "cpu".
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.
        """
        super().__init__(
            name="MLPClassifier",
            task_type="Supervised Classification",
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose,
        )
        self.device = device if device is not None else get_device()
        self.input_dim = None
        self.output_dim = None

    def _get_search_space(self) -> dict:
        """Define the hyperparameter search space for MLP classifier.

        Returns:
            dict: Dictionary describing the search space configuration with keys:
                - n_layers: Number of hidden layers (1-3)
                - hidden_dim_base: Base range for hidden layer sizes (32-512, log scale)
                - dropout: Dropout rate (0.0-0.5)
                - learning_rate: Learning rate (1e-4 to 1e-2, log scale)
                - batch_size: Batch size options [16, 32, 64, 128]
                - epochs: Number of training epochs (50-200)
        """
        return {
            "n_layers": {"type": "int", "low": 1, "high": 3},
            "hidden_dim_base": {"type": "int", "low": 32, "high": 512, "log": True},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "epochs": {"type": "int", "low": 50, "high": 200},
        }

    def create_model(self, trial: optuna.Trial):
        """Create MLP model with hyperparameters suggested by Optuna.

        Uses the search space defined in _get_search_space() to suggest hyperparameters
        for the model.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.

        Returns:
            PyTorchMLPWrapper: MLP model with trial-suggested hyperparameters.
        """
        search_space = self._get_search_space()

        # Number of hidden layers
        n_layers = trial.suggest_int(
            "n_layers",
            search_space["n_layers"]["low"],
            search_space["n_layers"]["high"],
        )

        # Hidden layer dimensions
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(
                f"hidden_dim_{i}",
                search_space["hidden_dim_base"]["low"],
                search_space["hidden_dim_base"]["high"],
                log=search_space["hidden_dim_base"]["log"],
            )
            hidden_dims.append(hidden_dim)

        # Dropout rate
        dropout = trial.suggest_float(
            "dropout", search_space["dropout"]["low"], search_space["dropout"]["high"]
        )

        # Learning rate
        learning_rate = trial.suggest_float(
            "learning_rate",
            search_space["learning_rate"]["low"],
            search_space["learning_rate"]["high"],
            log=search_space["learning_rate"]["log"],
        )

        # Batch size
        batch_size = trial.suggest_categorical(
            "batch_size", search_space["batch_size"]["choices"]
        )

        # Epochs
        epochs = trial.suggest_int(
            "epochs", search_space["epochs"]["low"], search_space["epochs"]["high"]
        )

        return PyTorchMLPWrapper(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.output_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            task_type="classification",
            device=self.device,
            early_stopping_patience=10,
        )

    def get_scoring_metric(self) -> str:
        """Return the scoring metric for classification.

        Returns:
            str: The scoring metric name ('accuracy').
        """
        return "accuracy"

    def _get_model_predictions(self, model, embeddings: np.ndarray):
        """Get probability predictions from the model.

        Args:
            model (PyTorchMLPWrapper): Trained MLP model.
            embeddings (np.ndarray): Input embeddings for prediction.

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, n_classes).
        """
        return model.predict_proba(embeddings)

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
    ) -> tuple:
        """Get predictions from the MLP classifier.

        Sets input/output dimensions based on the data before calling the parent's
        get_prediction method.

        Args:
            embeddings (np.ndarray): Input embeddings for prediction.
            y (np.ndarray | None, optional): Target labels for training. Required if train=True.
                Defaults to None.
            train (bool, optional): Whether to train the model. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - predictions (np.ndarray): Class probabilities.
                - additional_info (dict | None): Dictionary with optimization info if train=True,
                    None otherwise.

        Raises:
            ValueError: If train=True and y is None.
        """
        if train:
            if y is None:
                raise ValueError("y must be provided for training")

            # Set dimensions based on data
            self.input_dim = embeddings.shape[1]
            self.output_dim = len(np.unique(y))

        return super().get_prediction(embeddings, y, train)


class MLPRegressorEvaluator(AbstractHPOEvaluator):
    """MLP Regressor evaluator with Optuna hyperparameter optimization using PyTorch.

    This evaluator uses a PyTorch-based Multi-Layer Perceptron for regression
    tasks with automatic hyperparameter optimization via Optuna.

    Attributes:
        device (str): Device to run the model on ('cpu' or 'cuda').
        input_dim (int | None): Number of input features (set during training).
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize the MLPRegressorEvaluator.

        Args:
            n_trials (int, optional): Number of optimization trials. Defaults to 50.
            cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                Defaults to "cpu".
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.
        """
        super().__init__(
            name="MLPRegressor",
            task_type="Supervised Regression",
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose,
        )
        self.device = device if device is not None else get_device()
        self.input_dim = None

    def _get_search_space(self) -> dict:
        """Define the hyperparameter search space for MLP regressor.

        Returns:
            dict: Dictionary describing the search space configuration with keys:
                - n_layers: Number of hidden layers (1-3)
                - hidden_dim_base: Base range for hidden layer sizes (32-512, log scale)
                - dropout: Dropout rate (0.0-0.5)
                - learning_rate: Learning rate (1e-4 to 1e-2, log scale)
                - batch_size: Batch size options [16, 32, 64, 128]
                - epochs: Number of training epochs (50-200)
        """
        return {
            "n_layers": {"type": "int", "low": 1, "high": 3},
            "hidden_dim_base": {"type": "int", "low": 32, "high": 512, "log": True},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "epochs": {"type": "int", "low": 50, "high": 200},
        }

    def create_model(self, trial: optuna.Trial):
        """Create MLP model with hyperparameters suggested by Optuna.

        Uses the search space defined in _get_search_space() to suggest hyperparameters
        for the model.

        Args:
            trial (optuna.Trial): Optuna trial object for suggesting hyperparameters.

        Returns:
            PyTorchMLPWrapper: MLP model with trial-suggested hyperparameters.
        """
        search_space = self._get_search_space()

        # Number of hidden layers
        n_layers = trial.suggest_int(
            "n_layers",
            search_space["n_layers"]["low"],
            search_space["n_layers"]["high"],
        )

        # Hidden layer dimensions
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(
                f"hidden_dim_{i}",
                search_space["hidden_dim_base"]["low"],
                search_space["hidden_dim_base"]["high"],
                log=search_space["hidden_dim_base"]["log"],
            )
            hidden_dims.append(hidden_dim)

        # Dropout rate
        dropout = trial.suggest_float(
            "dropout", search_space["dropout"]["low"], search_space["dropout"]["high"]
        )

        # Learning rate
        learning_rate = trial.suggest_float(
            "learning_rate",
            search_space["learning_rate"]["low"],
            search_space["learning_rate"]["high"],
            log=search_space["learning_rate"]["log"],
        )

        # Batch size
        batch_size = trial.suggest_categorical(
            "batch_size", search_space["batch_size"]["choices"]
        )

        # Epochs
        epochs = trial.suggest_int(
            "epochs", search_space["epochs"]["low"], search_space["epochs"]["high"]
        )

        return PyTorchMLPWrapper(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,  # Single output for regression
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            task_type="regression",
            device=self.device,
            early_stopping_patience=10,
        )

    def get_scoring_metric(self) -> str:
        """Return the scoring metric for regression.

        Returns:
            str: The scoring metric name ('neg_mean_squared_error').
        """
        return "neg_mean_squared_error"

    def _get_model_predictions(self, model, embeddings: np.ndarray):
        """Get predictions from the model.

        Args:
            model (PyTorchMLPWrapper): Trained MLP model.
            embeddings (np.ndarray): Input embeddings for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        return model.predict(embeddings)

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y: np.ndarray | None = None,
        train: bool = True,
    ) -> tuple:
        """Get predictions from the MLP regressor.

        Sets input dimension based on the data before calling the parent's
        get_prediction method.

        Args:
            embeddings (np.ndarray): Input embeddings for prediction.
            y (np.ndarray | None, optional): Target values for training. Required if train=True.
                Defaults to None.
            train (bool, optional): Whether to train the model. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - predictions (np.ndarray): Predicted values.
                - additional_info (dict | None): Dictionary with optimization info if train=True,
                    None otherwise.

        Raises:
            ValueError: If train=True and y is None.
        """
        if train:
            if y is None:
                raise ValueError("y must be provided for training")

            # Set input dimension based on data
            self.input_dim = embeddings.shape[1]

        return super().get_prediction(embeddings, y, train)
