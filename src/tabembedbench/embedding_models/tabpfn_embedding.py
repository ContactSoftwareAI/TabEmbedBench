import logging
import warnings

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.utils import infer_categorical_features
from tabpfn_extensions.many_class import ManyClassClassifier
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device

class TabPFNEmbedding(AbstractEmbeddingGenerator):
    """Universal TabPFN-based embedding generator for tabular data.

    This class generates embeddings using TabPFN (Tabular Prior-data Fitted Networks)
    by treating each feature as a target and using the remaining features as inputs.
    It supports both classification and regression tasks automatically based on
    feature type detection.

    The embedding process works by:
    1. For each feature column, mask it as the target
    2. Use remaining features as inputs to predict the masked feature
    3. Extract embeddings from the trained TabPFN model
    4. Aggregate embeddings across features and estimators

    Attributes:
        num_estimators (int): Number of TabPFN estimators to use.
        estimator_agg (str): Aggregation method for multiple estimators.
        emb_agg (str): Aggregation method for feature embeddings.
        device (torch.device): Device for computation (CPU/GPU).
        tabpfn_dim (int): Dimensionality of TabPFN embeddings (192).
        categorical_indices (list[int]): Indices of categorical features.
        num_features (int): Number of features in the dataset.
    """

    def __init__(
        self,
        num_estimators: int = 1,
        estimator_agg: str = "mean",
        emb_agg: str = "mean",
    ) -> None:
        """Initialize the UniversalTabPFNEmbedding.

        Args:
            num_estimators (int, optional): Number of TabPFN estimators to use for
                ensemble predictions. Defaults to 1.
            estimator_agg (str, optional): Aggregation method for combining embeddings
                from multiple estimators. Options are "mean" or "first_element".
                Defaults to "mean".
            emb_agg (str, optional): Aggregation method for combining embeddings
                across features. Options are "concat" or "mean". Defaults to "mean".

        Raises:
            NotImplementedError: If unsupported aggregation methods are specified.
        """
        super().__init__(name="TabPFN")
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

        self.tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        self.tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)

        self._is_fitted = False

    def _preprocess_data(
        self, X: np.ndarray, train: bool = True, outlier: bool = False, **kwargs
    ):
        """Preprocess input data for TabPFN embedding generation.

        Converts numpy arrays to PyTorch tensors and moves them to the appropriate
        device (CPU/GPU) for computation.

        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features).
            train (bool, optional): Whether this is training data. Currently unused
                but kept for interface compatibility. Defaults to True.
            outlier (bool, optional): Whether this is outlier data. Currently unused
                but kept for interface compatibility. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Preprocessed data as a float tensor on the specified device.
        """
        numerical_transformer = TransformToNumerical()
        if outlier:
            X_preprocessed = numerical_transformer.fit_transform(X)
        else:
            train_indices = kwargs.get("train_indices")
            test_indices = kwargs.get("test_indices")

            if isinstance(X, pd.DataFrame):
                X_train = X.iloc[train_indices]
                X_test = X.iloc[test_indices]

                X_train = numerical_transformer.fit_transform(X_train)
                X_test = numerical_transformer.transform(X_test)
            else:
                X_train = X[train_indices]
                X_test = X[test_indices]

                X_train = numerical_transformer.fit_transform(X_train)
                X_test = numerical_transformer.transform(X_test)

            X_preprocessed = np.empty(X.values.shape, dtype=np.float64)

            X_preprocessed[train_indices] = X_train
            X_preprocessed[test_indices] = X_test

        X_preprocessed = self._handle_constant_features(X_preprocessed)

        return X_preprocessed.astype(np.float64)

    def _handle_constant_features(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Handle constant features by adding small noise to avoid numerical issues.

        Args:
            X (np.ndarray): Input data matrix.
            eps (float): Small epsilon value to add to constant features.

        Returns:
            np.ndarray: Data with constant features handled.
        """
        X = X.copy()

        # Check for constant columns (zero variance)
        feature_std = np.std(X, axis=0)
        constant_mask = feature_std < eps

        if np.any(constant_mask):
            self._logger.warning(
                f"Found {np.sum(constant_mask)} constant feature(s). "
                f"Adding small noise to avoid numerical issues."
            )
            # Add very small random noise to constant columns
            for col_idx in np.where(constant_mask)[0]:
                X[:, col_idx] += np.random.normal(0, eps, size=X.shape[0])

        return X

    def _fit_model(
        self,
        X_preprocessed: np.ndarray,
        categorical_indices: list[int] | None = None,
        **kwargs,
    ):
        """Fit the TabPFN embedding model to the preprocessed data.

        This method prepares the model for embedding generation by identifying
        categorical features and storing dataset metadata. No actual model training
        occurs here as TabPFN models are fitted during embedding computation.

        Args:
            X_preprocessed (torch.Tensor): Preprocessed input data of shape
                (n_samples, n_features).
            categorical_indices (list[int] | None, optional): List of indices
                indicating which features are categorical. If None, categorical
                features will be automatically inferred. Defaults to None.
            **kwargs: Additional keyword arguments (unused).

        Note:
            Sets the internal state to fitted and stores feature information
            for subsequent embedding computation.
        """
        if categorical_indices is not None:
            self.categorical_indices = categorical_indices
        else:
            # Convert CUDA tensor to CPU numpy array for categorical inference
            self.categorical_indices = infer_categorical_features(X_preprocessed)

        self.num_features = X_preprocessed.shape[-1]

        self._is_fitted = True

    def _compute_embeddings(
        self,
        X_preprocessed: np.ndarray,
        outlier: bool = False,
        **kwargs,
    ):
        """Compute embeddings using TabPFN models.

        For each feature column, this method:
        1. Masks the feature as the target variable
        2. Uses remaining features as inputs to train a TabPFN model
        3. Extracts embeddings from the trained model
        4. Aggregates embeddings according to the specified strategy

        The method automatically selects TabPFNClassifier for categorical features
        and TabPFNRegressor for continuous features.

        Args:
            X_preprocessed (np.ndarray): Preprocessed input data of shape
                (n_samples, n_features).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Generated embeddings. Shape depends on aggregation method:
                - If emb_agg="concat": (n_samples, n_features * tabpfn_dim)
                - If emb_agg="mean": (n_samples, tabpfn_dim)

        Raises:
            NotImplementedError: If unsupported aggregation methods are used.

        Note:
            This method creates fresh TabPFN models for each feature to avoid
            interference between different prediction tasks.
        """
        if outlier:
            X_embeddings = self._compute_internal_embeddings(
                X_preprocessed)

            return X_embeddings, None
        else:
            train_indices = kwargs.get("train_indices")
            test_indices = kwargs.get("test_indices")
            X_train = X_preprocessed[train_indices]

            X_train_embeddings = self._compute_internal_embeddings(X_train)

            X_embeddings = self._compute_internal_embeddings(X_preprocessed)

            X_test_embeddings = X_embeddings[test_indices]

            return X_train_embeddings, X_test_embeddings

    def _compute_internal_embeddings(self, X):
        num_samples = X.shape[0]
        tmp_embeddings = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
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
                    if "Unknown label type: continuous" in str(e):
                        self._logger.warning(
                            f"Using regression model for column {column_idx} due "
                            f"to the error: "
                            f"{str(e)}."
                        )
                        model = self.tabpfn_reg
                        model.fit(features, target)
                    elif "Number of classes" in str(e) and ("exceeds the maximal "
                                                                "number" in str(e)):
                        self._logger.warning(
                            f"Using regression model for column {column_idx} "
                            f"due to the error: "
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
        elif self.emb_agg == "mean":
            reshaped_embeddings = concat_embeddings.reshape(
                *concat_embeddings.shape[:-1], self.num_features, self.tabpfn_dim
            )
            embeddings = np.mean(reshaped_embeddings, axis=-2)

            return embeddings
        else:
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
        self.tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        self.tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)
        self.num_features = None
        self.categorical_indices = None
        self._is_fitted = False