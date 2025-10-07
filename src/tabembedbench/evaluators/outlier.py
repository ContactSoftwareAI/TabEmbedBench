import numpy as np
import math
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD

from tabembedbench.evaluators import AbstractEvaluator

class LocalOutlierFactorEvaluator(AbstractEvaluator):
    def __init__(
            self,
            model_params: dict,
    ):
        super().__init__(
            name="LocalOutlierFactor",
            task_type="Outlier Detection"
        )

        self.model_params = model_params

        self.lof = LocalOutlierFactor(
            **self.model_params
        )

    def get_prediction(
            self,
            embeddings: np.ndarray,
            y = None,
            train = True,
    ):
        self.lof.fit(embeddings)
        prediction = (-1)*self.lof.negative_outlier_factor_

        return prediction, None

    def reset_evaluator(self):
        self.lof = LocalOutlierFactor(
            **self.model_params
        )

    def get_parameters(self):
        return self.model_params


class ECODEvaluator(AbstractEvaluator):
    def __init__(
            self,
            model_params: dict = None,
    ):
        super().__init__(
            name="ECOD",
            task_type="Outlier Detection"
        )

        self.model_params = model_params or {}

        self.ecod = ECOD()

    def get_prediction(
            self,
            embeddings: np.ndarray,
            y = None,
            train = True,
    ):
        self.ecod.fit(embeddings)
        prediction = self.ecod.decision_function(embeddings)

        return prediction, None

    def reset_evaluator(self):
        self.ecod = ECOD()

    def get_parameters(self):
        return self.model_params


class DeepSVDDEvaluator(AbstractEvaluator):
    def __init__(
            self,
            dynamic_hidden_neurons: bool = False,
            use_ae: bool = False,
            hidden_neurons: list = [64, 32],
            hidden_activation: str = "relu",
            output_activation: str = 'sigmoid',
            optimizer: str = "adam",
            epochs: int = 200,
            batch_size: int = 32,
            dropout_rate: float = 0.2,
            l2_regularizer: float = 0.1,
            validation_size: float = 0.1,
            preprocessing: bool = True,
            contamination: float = 0.1,
            random_seed: int = 42
    ):
        super().__init__(
            name="DeepSVDD",
            task_type="Outlier Detection"
        )

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
            "random_state": random_seed
        }
        self._validate_parameters(self.model_params)
        self._current_model_params = None

        self.random_seed = random_seed

        self.deep_svdd = None

    @staticmethod
    def _validate_parameters(model_params: dict):
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
        """
        Compute a list of hidden neuron sizes starting from the closest power of two
        or 64 of n_features/2, going down to 32.

        Args:
            n_features: Number of input features

        Returns:
            List of hidden neuron sizes (powers of 2, from computed value down to 32)
        """
        half_features = n_features / 2

        # If half_features <= 64, start from 64
        if half_features <= 64:
            start_power = 64
        else:
            log_val = math.log2(half_features)

            # Get the floor and ceil powers of 2
            lower_power = 2 ** int(log_val)
            upper_power = 2 ** (int(log_val) + 1)

            # Choose the closest one
            if (half_features - lower_power) < (upper_power - half_features):
                start_power = max(64, lower_power)
            else:
                start_power = max(64, upper_power)

        # Generate list from start_power down to 32 (powers of 2)
        result = []
        current = start_power
        while current >= 32:
            result.append(current)
            current = current // 2

        return result

    def get_prediction(
            self,
            embeddings: np.ndarray,
            y = None,
            train = True,
    ):
        model_params = self.model_params.copy()
        if self.deep_svdd is None:
            n_features = embeddings.shape[1]
            model_params["n_features"] = n_features
            if self.dynamic_hidden_neurons:
                hidden_neurons = self._compute_hidden_neurons(n_features)
                model_params["hidden_neurons"] = hidden_neurons
            self.deep_svdd = DeepSVDD(
                **model_params
            )
            self._current_model_params = model_params
        self.deep_svdd.fit(embeddings)
        prediction = self.deep_svdd.decision_function(embeddings)

        return prediction, None

    def reset_evaluator(self):
        self.deep_svdd = None
        self._current_model_params = None

    def get_parameters(self) -> dict:
        parameters = {
            key: value for key, value in self._current_model_params.items()
            if key != "hidden_neurons"
        }

        parameters["dynamic_hidden_neurons"] = self.dynamic_hidden_neurons

        return parameters


class IsolationForestEvaluator(AbstractEvaluator):
    def __init__(
            self,
            model_params: dict,
            random_seed: int = 42
    ):
        super().__init__(
            name="IsolationForest",
            task_type="Outlier Detection")

        self.model_params = model_params
        self.random_seed = random_seed

        self.iso_forest = IsolationForest(
            **self.model_params,
            random_state=self.random_seed
        )

    def get_prediction(
            self,
            embeddings: np.ndarray,
            y = None,
            train = True,
    ):
        self.iso_forest.fit(embeddings)
        prediction = (-1)*self.iso_forest.decision_function(embeddings)

        return prediction, None

    def reset_evaluator(self):
        self.iso_forest = IsolationForest(
            **self.model_params,
            random_state=self.random_seed
        )

    def get_parameters(self):
        return self.model_params
