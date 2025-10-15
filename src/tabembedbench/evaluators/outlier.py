import numpy as np
import math
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD

from tabembedbench.evaluators import AbstractEvaluator


class LocalOutlierFactorEvaluator(AbstractEvaluator):
    """Local Outlier Factor evaluator for outlier detection in embeddings.

    This evaluator uses scikit-learn's LocalOutlierFactor algorithm to detect
    outliers in embedding space. LOF computes the local density deviation of a
    given data point with respect to its neighbors.

    Attributes:
        model_params (dict): Dictionary containing LOF model parameters such as
            n_neighbors, metric, etc.
        lof (LocalOutlierFactor): The underlying scikit-learn LOF model.
    """

    def __init__(
        self,
        model_params: dict,
    ):
        """Initialize the Local Outlier Factor evaluator.

        Args:
            model_params (dict): Parameters to pass to LocalOutlierFactor.
                Common parameters include:
                - n_neighbors (int): Number of neighbors to use.
                - metric (str): Distance metric to use.
                - contamination (float): Expected proportion of outliers.
        """
        super().__init__(name="LocalOutlierFactor", task_type="Outlier Detection")

        self.model_params = model_params

        self.lof = LocalOutlierFactor(**self.model_params)

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y=None,
        train=True,
    ):
        """Get outlier scores from the LOF model.

        Args:
            embeddings (np.ndarray): Input embeddings of shape (n_samples, n_features).
            y: Unused parameter, kept for interface compatibility. Defaults to None.
            train: Unused parameter, kept for interface compatibility. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - prediction (np.ndarray): Outlier scores where higher values indicate
                    more likely outliers. Shape (n_samples,).
                - additional_info (None): No additional information is returned.
        """
        self.lof.fit(embeddings)
        prediction = (-1) * self.lof.negative_outlier_factor_

        return prediction, None

    def reset_evaluator(self):
        """Reset the evaluator to its initial state.

        Reinitializes the LOF model with the original parameters, clearing
        any fitted state.
        """
        self.lof = LocalOutlierFactor(**self.model_params)

    def get_parameters(self):
        """Get the current parameters of the evaluator.

        Returns:
            dict: Dictionary containing all LOF model parameters.
        """
        return self.model_params


class DeepSVDDEvaluator(AbstractEvaluator):
    """Deep Support Vector Data Description evaluator for outlier detection.

    This evaluator uses the Deep SVDD algorithm from PyOD for outlier detection
    in embeddings. Deep SVDD trains a neural network to map data into a hypersphere
    and identifies outliers as points far from the center.

    Attributes:
        dynamic_hidden_neurons (bool): Whether to dynamically compute hidden layer sizes
            based on input dimensionality.
        model_params (dict): Dictionary containing Deep SVDD model parameters.
        random_seed (int): Random seed for reproducibility.
        deep_svdd (DeepSVDD | None): The underlying PyOD Deep SVDD model.
        _current_model_params (dict | None): Current model parameters after fitting.
    """

    def __init__(
        self,
        dynamic_hidden_neurons: bool = False,
        use_ae: bool = False,
        hidden_neurons: list = [64, 32],
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
        optimizer: str = "adam",
        epochs: int = 250,
        batch_size: int = 32,
        dropout_rate: float = 0.2,
        l2_regularizer: float = 0.1,
        validation_size: float = 0.1,
        preprocessing: bool = True,
        contamination: float = 0.1,
        random_seed: int = 42,
    ):
        """Initialize the Deep SVDD evaluator.

        Args:
            dynamic_hidden_neurons (bool, optional): If True, automatically computes
                hidden layer sizes based on input features. Defaults to False.
            use_ae (bool, optional): Whether to use autoencoder for pretraining.
                Defaults to False.
            hidden_neurons (list, optional): List of hidden layer sizes. Defaults to [64, 32].
            hidden_activation (str, optional): Activation function for hidden layers.
                Defaults to "relu".
            output_activation (str, optional): Activation function for output layer.
                Defaults to 'sigmoid'.
            optimizer (str, optional): Optimizer to use. Defaults to "adam".
            epochs (int, optional): Number of training epochs. Defaults to 200.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
            l2_regularizer (float, optional): L2 regularization strength. Defaults to 0.1.
            validation_size (float, optional): Proportion of data for validation.
                Defaults to 0.1.
            preprocessing (bool, optional): Whether to apply preprocessing. Defaults to True.
            contamination (float, optional): Expected proportion of outliers. Defaults to 0.1.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Raises:
            ValueError: If any parameter values are invalid.
        """
        super().__init__(name="DeepSVDD", task_type="Outlier Detection")

        self.dynamic_hidden_neurons = dynamic_hidden_neurons

        self.model_params = {
            "use_ae": use_ae,
            "hidden_neurons": hidden_neurons,
            "hidden_activation": hidden_activation,
            "output_activation": output_activation,
            "optimizer": optimizer,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "l2_regularizer": l2_regularizer,
            "validation_size": validation_size,
            "preprocessing": preprocessing,
            "contamination": contamination,
            "random_state": random_seed,
        }
        self._validate_parameters(self.model_params)
        self._current_model_params = None

        self.random_seed = random_seed

        self.deep_svdd = None

    @staticmethod
    def _validate_parameters(model_params: dict):
        """Validate Deep SVDD model parameters.

        Args:
            model_params (dict): Dictionary of model parameters to validate.

        Returns:
            bool: True if all parameters are valid.

        Raises:
            ValueError: If any parameter value is invalid.
        """
        for key, value in model_params.items():
            if key == "dropout_rate":
                if value < 0 or value > 1:
                    raise ValueError("Dropout rate is invalid")
            elif key == "l2_reqularizer":
                if value < 0:
                    raise ValueError("L2 regularization is invalid")
            elif key == "validation_size":
                if value < 0 or value > 1:
                    raise ValueError("Validation size is invalid")
            elif key == "contamination":
                if value < 0 or value > 0.5:
                    raise ValueError("Contamination is invalid")
        return True

    @staticmethod
    def _compute_hidden_neurons(n_features: int) -> list:
        """Compute hidden layer sizes dynamically based on input features.

        Creates a decreasing sequence of hidden layer sizes starting from a power
        of 2 close to half the number of features, down to 32.

        Args:
            n_features (int): Number of input features.

        Returns:
            list: List of hidden layer sizes in decreasing order.
        """
        half_features = n_features / 2

        if half_features <= 64:
            start_power = 64
        else:
            log_val = math.log2(half_features)

            lower_power = 2 ** int(log_val)
            upper_power = 2 ** (int(log_val) + 1)

            if (half_features - lower_power) < (upper_power - half_features):
                start_power = min(64, lower_power)
            else:
                start_power = min(64, upper_power)

        result = []
        current = start_power
        while current >= 32:
            result.append(current)
            current = current // 2

        return result

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y=None,
        train=True,
    ):
        """Get outlier scores from the Deep SVDD model.

        Args:
            embeddings (np.ndarray): Input embeddings of shape (n_samples, n_features).
            y: Unused parameter, kept for interface compatibility. Defaults to None.
            train: Unused parameter, kept for interface compatibility. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - prediction (np.ndarray): Outlier scores where higher values indicate
                    more likely outliers. Shape (n_samples,).
                - additional_info (None): No additional information is returned.
        """
        model_params = self.model_params.copy()
        if self.deep_svdd is None:
            n_features = embeddings.shape[1]
            model_params["n_features"] = n_features
            if self.dynamic_hidden_neurons:
                hidden_neurons = self._compute_hidden_neurons(n_features)
                model_params["hidden_neurons"] = hidden_neurons
            self.deep_svdd = DeepSVDD(**model_params)
            self._current_model_params = model_params
        self.deep_svdd.fit(embeddings)
        prediction = self.deep_svdd.decision_function(embeddings)

        return prediction, None

    def reset_evaluator(self):
        """Reset the evaluator to its initial state.

        Clears the fitted Deep SVDD model and current model parameters.
        """
        self.deep_svdd = None
        self._current_model_params = None

    def get_parameters(self) -> dict:
        """Get the current parameters of the evaluator.

        Returns:
            dict: Dictionary containing all Deep SVDD model parameters, excluding
                the hidden_neurons list but including dynamic_hidden_neurons flag.
        """
        parameters = {
            key: value
            for key, value in self._current_model_params.items()
            if key != "hidden_neurons"
        }

        parameters["dynamic_hidden_neurons"] = self.dynamic_hidden_neurons

        return parameters


class IsolationForestEvaluator(AbstractEvaluator):
    """Isolation Forest evaluator for outlier detection in embeddings.

    This evaluator uses scikit-learn's IsolationForest algorithm to detect outliers
    in embedding space. Isolation Forest isolates observations by randomly selecting
    a feature and then randomly selecting a split value between the maximum and
    minimum values of the selected feature.

    Attributes:
        model_params (dict): Dictionary containing Isolation Forest model parameters.
        random_seed (int): Random seed for reproducibility.
        iso_forest (IsolationForest): The underlying scikit-learn Isolation Forest model.
    """

    def __init__(self, model_params: dict, random_seed: int = 42):
        """Initialize the Isolation Forest evaluator.

        Args:
            model_params (dict): Parameters to pass to IsolationForest.
                Common parameters include:
                - n_estimators (int): Number of trees in the forest.
                - max_samples (int or float): Number of samples to draw.
                - contamination (float): Expected proportion of outliers.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__(name="IsolationForest", task_type="Outlier Detection")

        self.model_params = model_params
        self.random_seed = random_seed

        self.iso_forest = IsolationForest(
            **self.model_params, random_state=self.random_seed
        )

    def get_prediction(
        self,
        embeddings: np.ndarray,
        y=None,
        train=True,
    ):
        """Get outlier scores from the Isolation Forest model.

        Args:
            embeddings (np.ndarray): Input embeddings of shape (n_samples, n_features).
            y: Unused parameter, kept for interface compatibility. Defaults to None.
            train: Unused parameter, kept for interface compatibility. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - prediction (np.ndarray): Outlier scores where higher values indicate
                    more likely outliers. Shape (n_samples,).
                - additional_info (None): No additional information is returned.
        """
        self.iso_forest.fit(embeddings)
        prediction = (-1) * self.iso_forest.decision_function(embeddings)

        return prediction, None

    def reset_evaluator(self):
        """Reset the evaluator to its initial state.

        Reinitializes the Isolation Forest model with the original parameters,
        clearing any fitted state.
        """
        self.iso_forest = IsolationForest(
            **self.model_params, random_state=self.random_seed
        )

    def get_parameters(self):
        """Get the current parameters of the evaluator.

        Returns:
            dict: Dictionary containing all Isolation Forest model parameters.
        """
        return self.model_params
