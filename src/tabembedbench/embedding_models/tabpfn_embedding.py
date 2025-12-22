import gc
import warnings

import numpy as np
import pandas as pd
import polars as pl
from tabicl.sklearn.preprocessing import TransformToNumerical
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.utils import infer_categorical_features

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device

MAX_UNIQUE_FOR_CATEGORICAL_FEATURES = 30
MIN_UNIQUE_FOR_NUMERICAL_FEATURES = 4
MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE = 100


class TabPFNEmbedding(AbstractEmbeddingGenerator):
    """TabPFN-based embedding generator for tabular data.

    This class generates embeddings using the TabPFN (Tabular Prior-Fitted Network)
    foundation model. It employs a unique self-supervised approach where each column
    is treated alternately as a target variable while other columns serve as features.
    This column-wise prediction strategy captures rich feature interactions and
    relationships within the tabular data.

    The embedding generation process:
    1. For each column, treat it as the target and remaining columns as features
    2. Fit a TabPFN classifier (for categorical) or regressor (for numerical) model
    3. Extract embeddings from the fitted model
    4. Aggregate embeddings across columns and estimators

    Attributes:
        num_estimators (int): Number of estimators used for ensemble predictions.
        device (str): Device for computation ('cpu' or 'cuda').
        tabpfn_dim (int): Dimensionality of TabPFN embeddings (192).
        emb_agg (str): Method for aggregating embeddings across columns ('mean' or 'concat').
        estimator_agg (str): Method for aggregating outputs from multiple estimators.
        tabpfn_clf (TabPFNClassifier): TabPFN classifier instance for categorical targets.
        tabpfn_reg (TabPFNRegressor): TabPFN regressor instance for numerical targets.
        num_features (int | None): Number of features in the input data.
        categorical_indices (list[int] | None): Indices of categorical features.
    """

    def __init__(
        self,
        num_estimators: int = 1,
        estimator_agg: str = "mean",
        emb_agg: str = "mean",
    ) -> None:
        """Initialize the TabPFN embedding generator.

        Args:
            num_estimators (int, optional): Number of estimators for ensemble predictions.
                Defaults to 1.
            estimator_agg (str, optional): Method for aggregating estimator outputs.
                Options: 'mean', 'first_element'. Defaults to "mean".
            emb_agg (str, optional): Method for aggregating embeddings across columns.
                Options: 'mean', 'concat'. Defaults to "mean".
        """
        super().__init__(
            name="TabPFN Embedding",
            max_num_samples=15000,
            max_num_features=500,
        )
        self.num_estimators = num_estimators

        self.device = get_device()

        self.tabpfn_dim = 192

        self._init_tabpfn_configs = {
            "device": self.device,
            "n_estimators": self.num_estimators,
            "ignore_pretraining_limits": True,
            "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
        }

        self.emb_agg = emb_agg
        self.estimator_agg = estimator_agg

        self.tabpfn_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        self.tabpfn_reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)

        self.transform_to_numerical = None

        self._is_fitted = False

    def _preprocess_data(
        self,
        X: np.ndarray | pd.DataFrame,
        train: bool = True,
        outlier: bool = False,
        **kwargs,
    ) -> pl.DataFrame:
        """Preprocess input data by converting to float64.

        Args:
            X (np.ndarray): Input data to preprocess.
            train (bool, optional): Whether this is training mode. Defaults to True.
            outlier (bool, optional): Whether to handle outliers. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Data converted to float64 dtype.
        """
        return X

    def _fit_model(
        self,
        X_preprocessed: np.ndarray | pd.DataFrame,
        categorical_indices: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Fit the model by identifying categorical features.

        This method prepares the model for embedding generation by determining
        which features are categorical (either from provided indices or by inference)
        and storing the number of features.

        Args:
            X_preprocessed (np.ndarray): Preprocessed input data.
            categorical_indices (list[int] | None, optional): Indices of categorical
                features. If None, categorical features are automatically inferred.
                Defaults to None.
            **kwargs: Additional keyword arguments (unused).
        """
        if categorical_indices is not None:
            self.categorical_indices = categorical_indices
        else:
            self.categorical_indices = infer_categorical_features(
                X_preprocessed,
                provided=None,
                min_samples_for_inference=MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
                max_unique_for_category=MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
                min_unique_for_numerical=MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
            )

        self.num_features = X_preprocessed.shape[-1]

        self._is_fitted = True

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray | pd.DataFrame,
        X_test_preprocessed: np.ndarray | pd.DataFrame | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute embeddings using the column-wise TabPFN approach.

        In standard mode, computes embeddings for both train and test data by stacking
        them together to ensure consistent feature representations. In outlier mode,
        only training data embeddings are computed.

        Args:
            X_train_preprocessed (np.ndarray): Preprocessed training dataset.
            X_test_preprocessed (np.ndarray | None, optional): Preprocessed test dataset.
                Required when outlier is False. Defaults to None.
            outlier (bool, optional): If True, computes embeddings only for training data.
                Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            tuple[np.ndarray, np.ndarray | None]: A tuple containing:
                - train_embeddings: Embeddings for training data
                - test_embeddings: Embeddings for test data, or None if outlier is True.
        """
        if outlier:
            X_embeddings = self._compute_embeddings_per_columns(X_train_preprocessed)

            return X_embeddings, None
        X_train_embeddings = self._compute_embeddings_per_columns(X_train_preprocessed)

        size_X_train = X_train_preprocessed.shape[0]

        X_train_test_stack = np.vstack([X_train_preprocessed, X_test_preprocessed])

        X_embeddings = self._compute_embeddings_per_columns(X_train_test_stack)

        X_test_embeddings = X_embeddings[size_X_train:]

        return X_train_embeddings, X_test_embeddings

    def _compute_embeddings_with_random_vector(
        self, X_train: np.ndarray, X_test: np.ndarray | None
    ) -> np.ndarray:
        """Compute embeddings with random vector."""
        pass

    def _compute_embeddings_per_columns(self, X: np.ndarray) -> np.ndarray:
        """Compute embeddings by treating each column as a prediction target.

        This method implements the core column-wise embedding strategy. For each column:
        1. Treat the column as the target variable
        2. Use remaining columns as features
        3. Fit appropriate TabPFN model (classifier for categorical, regressor for numerical)
        4. Extract embeddings from the fitted model
        5. Aggregate across estimators and columns

        Args:
            X (np.ndarray): Input matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Aggregated embeddings. Shape depends on aggregation method:
                - If emb_agg='mean': (n_samples, tabpfn_dim)
                - If emb_agg='concat': (n_samples, n_features * tabpfn_dim)

        Raises:
            ValueError: If TabPFN model cannot be fitted to a column.
            NotImplementedError: If an unsupported aggregation method is specified.

        Note:
            Automatically falls back to regression model if a column marked as
            categorical contains continuous values or too many classes.
        """
        num_samples = X.shape[0]
        tmp_embeddings = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for column_idx in range(X.shape[1]):
                # Create mask for the current column
                mask = np.zeros_like(X, dtype=bool)
                mask[:, column_idx] = True

                # Extract features (all columns except current) and target (current column)
                features = X[~mask].reshape(num_samples, -1)
                target = X[mask]

                model = (
                    self.tabpfn_clf
                    if column_idx in self.categorical_indices
                    else self.tabpfn_reg
                )

                try:
                    model.fit(features, target)
                except ValueError as e:
                    # If a column is marked as categorical but has continuous values,
                    # fall back to using the regression model
                    if (
                        "Unknown label type: continuous" in str(e)
                        or "Number of classes" in str(e)
                        and ("exceeds the maximal number" in str(e))
                    ):
                        self._logger.warning(
                            f"Using regression model for column {column_idx} due "
                            f"to the error: "
                            f"{str(e)}."
                        )
                        model = self.tabpfn_reg
                        model.fit(features, target)
                    else:
                        raise ValueError("Can't fit TabPFN model.")

                estimator_embeddings = model.get_embeddings(features)

                if self.num_estimators > 1:
                    if self.estimator_agg == "mean":
                        estimator_embeddings = np.mean(estimator_embeddings, axis=0)
                    elif self.estimator_agg == "first_element":
                        estimator_embeddings = np.squeeze(estimator_embeddings[0, :])
                    else:
                        raise NotImplementedError
                else:
                    estimator_embeddings = np.squeeze(estimator_embeddings)

                tmp_embeddings += [estimator_embeddings]

        concat_embeddings = np.concatenate(tmp_embeddings, axis=1).reshape(
            tmp_embeddings[0].shape[0], -1
        )

        if self.emb_agg == "concat":
            return concat_embeddings
        if self.emb_agg == "mean":
            reshaped_embeddings = concat_embeddings.reshape(
                *concat_embeddings.shape[:-1], self.num_features, self.tabpfn_dim
            )
            embeddings = np.mean(reshaped_embeddings, axis=-2)

            return embeddings
        raise NotImplementedError

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        This method reinitializes the TabPFN models and clears all fitted state,
        allowing the model to be used on new datasets. It's typically called
        between different datasets or experiments.

        Note:
            After calling this method, the model will need to be fitted again
            before generating embeddings.
        """
        # Delete old model references
        del self.tabpfn_clf
        del self.tabpfn_reg

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if using CUDA or MPS
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()
        elif self.device == "mps":
            import torch

            torch.mps.empty_cache()

        # Reinitialize models
        self.tabpfn_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        self.tabpfn_reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        self.num_features = None
        self.categorical_indices = None
        self._is_fitted = False
        self.transform_to_numerical = None


class TabPFNWrapper(TabPFNEmbedding):
    """
    Wrapper class to handle TabPFN for predictive modeling.

    This class extends the functionality provided by TabPFNEmbedding, allowing
    the use of supervised classification or regression tasks with TabPFN as the
    underlying model. The wrapper offers the ability to fit a model to data,
    generate predictions, and reset the state of the embedding model. Instances
    of this class are designed to interface seamlessly with TabPFN models.

    Attributes:
        task_model (object | None): The specific TabPFN model instance being used
            for the current task (classification or regression). None if the model
            has not been initialized.
    """

    def __init__(
        self,
        num_estimators: int = 1,
    ) -> None:
        super().__init__(num_estimators=num_estimators)
        self._is_end_to_end_model = True
        self.name = "TabPFN"
        self.compatible_tasks_for_end_to_end = TABARENA_TASKS
        self.task_model = None

    def _fit_model(
        self,
        X_preprocessed: np.ndarray,
        y_preprocessed: np.ndarray | None = None,
        task_type=SUPERVISED_BINARY_CLASSIFICATION,
        **kwargs,
    ):
        self.task_model = (
            self.tabpfn_clf
            if task_type
            in (
                "Supervised Binary Classification",
                "Supervised Multiclass Classification",
            )
            else self.tabpfn_reg
        )
        self.task_model.fit(X=X_preprocessed, y=y_preprocessed)
        self._is_fitted = True

    def _get_prediction(
        self,
        X: np.ndarray,
        task_type: str = SUPERVISED_BINARY_CLASSIFICATION,
    ) -> np.ndarray:
        if not self.task_model:
            raise ValueError("Something went wrong.")
        if task_type in CLASSIFICATION_TASKS:
            probs = self.task_model.predict_proba(X)
            if task_type == SUPERVISED_BINARY_CLASSIFICATION and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        return self.task_model.predict(X)

    def _reset_embedding_model(self):
        super()._reset_embedding_model()
        self.task_model = None
