import gc
import warnings
from asyncio.unix_events import BaseChildWatcher
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
from tabicl.sklearn.preprocessing import TransformToNumerical
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.utils import infer_categorical_features
from umap import UMAP

from tabembedbench.constants import (
    CLASSIFICATION_TASKS,
    SUPERVISED_BINARY_CLASSIFICATION,
    TABARENA_TASKS,
)
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.embedding_models.tfm_helper import (
    get_cluster_distance_targets,
    merge_clusters_to_max_k,
)
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
        self.tabpfn_clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, **self._init_tabpfn_configs
        )
        self.tabpfn_reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, **self._init_tabpfn_configs
        )

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
        if isinstance(X, np.ndarray):
            if X.dtype.kind in "OSU":  # Object, String, Unicode
                X_df = pd.DataFrame(X)
            else:
                return X.astype(np.float64)
        else:
            X_df = X.copy()

        # Factorize object/string columns to numeric
        for col in X_df.select_dtypes(include=["object", "category", "string"]).columns:
            X_df[col] = pd.factorize(X_df[col])[0]

        return X_df.values.astype(np.float64)

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
            if not isinstance(X_preprocessed, np.ndarray):
                if hasattr(X_preprocessed, "to_numpy"):
                    X_preprocessed = X_preprocessed.to_numpy()
                elif hasattr(X_preprocessed, "values"):
                    X_preprocessed = X_preprocessed.values

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
        size_X_train = X_train_preprocessed.shape[0]
        if outlier:
            X_embeddings = self._compute_embeddings_internal(X_train_preprocessed)

            return X_embeddings, None
        X_train_embeddings = self._compute_embeddings_internal(X_train_preprocessed)

        X_train_test_stack = np.vstack([X_train_preprocessed, X_test_preprocessed])

        X_embeddings = self._compute_embeddings_internal(X_train_test_stack)

        X_test_embeddings = X_embeddings[size_X_train:]

        return X_train_embeddings, X_test_embeddings

    def _compute_embeddings_internal(
        self, X: NDArray, targets: NDArray | None = None
    ) -> np.ndarray:
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
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

        # Reinitialize models
        self.tabpfn_clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, **self._init_tabpfn_configs
        )
        self.tabpfn_reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, **self._init_tabpfn_configs
        )
        self.num_features = None
        self.categorical_indices = None
        self._is_fitted = False
        self.transform_to_numerical = None


class TabPFNEmbeddingConstantVector(TabPFNEmbedding):
    def __init__(self, num_estimators: int = 1):
        super().__init__(
            num_estimators=num_estimators,
        )
        self.name = "TabPFN with Constant Vector"

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray | pd.DataFrame,
        X_test_preprocessed: np.ndarray | pd.DataFrame | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        train_size = X_train_preprocessed.shape[0]
        train_targets = np.zeros((train_size,))

        X_train_embedded = self._compute_embeddings_internal(
            X_train_preprocessed, train_targets
        )

        if X_test_preprocessed is None:
            return X_train_embedded, None

        test_size = X_test_preprocessed.shape[0]
        test_targets = np.zeros((test_size,))

        X_train_test_stack = np.vstack([X_train_preprocessed, X_test_preprocessed])
        targets_stack = np.concatenate([train_targets, test_targets])

        X_train_test_embedded = self._compute_embeddings_internal(
            X_train_test_stack, targets_stack
        )

        return X_train_embedded, X_train_test_embedded[train_size:]

    def _compute_embeddings_internal(
        self, X: NDArray, targets: NDArray | None = None
    ) -> np.ndarray:
        model = self.tabpfn_clf
        model.fit(X, targets)

        embeddings = model.get_embeddings(X)

        if self.num_estimators > 1:
            if self.estimator_agg == "mean":
                embeddings = np.mean(embeddings, axis=0)
            elif self.estimator_agg == "first_element":
                embeddings = np.squeeze(embeddings[0, :])
            else:
                raise NotImplementedError
        else:
            embeddings = np.squeeze(embeddings)

        return embeddings


class TabPFNEmbeddingRandomVector(TabPFNEmbedding):
    DISCRETE_DISTRIBUTION = ["integers", "binomial", "poisson", "geometric"]

    def __init__(
        self,
        num_estimators: int = 1,
        random_state=None,
        epsilon: float = 1e-7,
        num_targets: int = 1,
        distribution: str = "standard_normal",
        dist_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            num_estimators=num_estimators,
        )
        self.epsilon = epsilon
        self.distribution = distribution
        self.dist_kwargs = dist_kwargs or {}
        self.name = f"TabPFN with Random Vector ({distribution})"
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.num_targets = num_targets
        self.random_train_targets = None
        self.is_discrete = self.distribution in self.DISCRETE_DISTRIBUTION

    def _generate_random_targets(self, size: tuple[int, int]) -> np.ndarray:
        try:
            dist_func = getattr(self.rng, self.distribution)

            return dist_func(size=size, **self.dist_kwargs)
        except AttributeError:
            raise ValueError(
                f"NumPy Generator has no distribution '{self.distribution}'."
            )

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray | pd.DataFrame,
        X_test_preprocessed: np.ndarray | pd.DataFrame | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        train_size = X_train_preprocessed.shape[0]
        train_targets = self._generate_random_targets(
            size=(train_size, self.num_targets)
        )

        X_train_embedded = self._compute_embeddings_internal(
            X_train_preprocessed, train_targets
        )

        if X_test_preprocessed is None:
            return X_train_embedded, None

        test_size = X_test_preprocessed.shape[0]
        test_targets = self._generate_random_targets(size=(test_size, self.num_targets))

        X_train_test_stack = np.vstack([X_train_preprocessed, X_test_preprocessed])
        targets_stack = np.vstack([train_targets, test_targets])

        X_train_test_embedded = self._compute_embeddings_internal(
            X_train_test_stack, targets_stack
        )

        return X_train_embedded, X_train_test_embedded[train_size:]

    def _compute_embeddings_internal(
        self, X: NDArray, targets: NDArray | None = None
    ) -> np.ndarray:
        target_embeddings = []
        num_targets = targets.shape[-1] if len(targets.shape) > 1 else 1
        model = self.tabpfn_clf if self.is_discrete else self.tabpfn_reg

        for i in range(num_targets):
            target = targets[:, i]
            model.fit(X, target)

            tmp_embeddings = model.get_embeddings(X)

            if self.num_estimators > 1:
                if self.estimator_agg == "mean":
                    tmp_embeddings = np.mean(tmp_embeddings, axis=0)
                elif self.estimator_agg == "first_element":
                    tmp_embeddings = np.squeeze(tmp_embeddings[0, :])
                else:
                    raise NotImplementedError
            else:
                tmp_embeddings = np.squeeze(tmp_embeddings)

            target_embeddings.append(tmp_embeddings)

        concat_embeddings = np.concatenate(target_embeddings, axis=1).reshape(
            target_embeddings[0].shape[0], -1
        )

        if self.emb_agg == "concat":
            return concat_embeddings
        if self.emb_agg == "mean":
            reshaped_embeddings = concat_embeddings.reshape(
                *concat_embeddings.shape[:-1], self.num_targets, self.tabpfn_dim
            )
            embeddings = np.mean(reshaped_embeddings, axis=-2)

            return embeddings
        raise NotImplementedError

    def _reset_embedding_model(self):
        super()._reset_embedding_model()
        self.rng = np.random.default_rng(self.random_state)
        self.random_train_targets = None


class TabPFNEmbeddingClusterLabels(TabPFNEmbedding):
    def __init__(
        self,
        clusterer: Optional[BaseEstimator] = None,
        table_encoder: Optional[TransformerMixin] = None,
        classifier: Optional[BaseEstimator] = None,
        mode: Literal["auto", "classification", "regression"] = "auto",
        distance_metrics: Optional[list[str]] = None,
        max_clusters: int = 10,
        num_estimators: int = 1,
        random_state: int = None,
    ):
        """
        TabPFN embedding using cluster-derived targets.

        Args:
            clusterer: Clustering algorithm (e.g., HDBSCAN, KMeans)
            table_encoder: Encoder to convert tables to numerical format
                (default: TableVectorizer)
            classifier: Classifier for inductive predictions
                (default: KNeighborsClassifier with distance weighting)
            mode: How to generate targets:
                - "auto": Use classification if ≤max_clusters, else regression
                - "classification": Always use cluster labels
                - "regression": Always use distance-based targets
            distance_metrics: Which distance metrics to use in regression mode
                Options: ["min", "mean", "std"] (default: ["min"])
            max_clusters: Maximum clusters for classification mode
            num_estimators: Number of TabPFN ensemble members
        """
        super().__init__(num_estimators=num_estimators)
        self.clusterer = clusterer
        self.table_encoder = table_encoder
        self.classifier = classifier
        self.mode = mode
        self.distance_metrics = distance_metrics or ["min"]
        self.max_clusters = max_clusters
        self.name = f"TabPFN with Cluster Labels ({mode})"

        self.actual_mode_: Optional[str] = None
        self.n_clusters_: Optional[int] = None
        self.random_state = random_state

    def get_hdbscan_pipeline(self, num_samples, num_features) -> HDBSCAN | Pipeline:
        """
        Generates an HDBSCAN clustering pipeline for the provided dataset dimensions and
        sample size, with optional dimensionality reduction based on feature count.

        The pipeline incorporates optional dimensionality reduction using UMAP
        if the feature count exceeds a specified threshold. The clustering is then
        performed using the HDBSCAN algorithm, with parameters tailored based on
        data characteristics. This method ensures the scalability of clustering
        and adjusts configurations accordingly for effective performance.

        Args:
            num_samples: int. The number of data samples in the dataset.
            num_features: int. The number of features (dimensionality) in the dataset.

        Returns:
            Pipeline or HDBSCAN: The constructed pipeline comprising UMAP (for optional
            dimensionality reduction) followed by HDBSCAN clustering, or an HDBSCAN
            instance when dimensionality reduction is not required.

        """
        needs_reduction = num_features > 50

        if needs_reduction:
            if num_features <= 100:
                effective_dims = 30  # Mild reduction for moderately high-dim
            elif num_features <= 300:
                effective_dims = 50  # Still within HDBSCAN's comfort zone
            elif num_features <= 1000:
                effective_dims = min(50, num_features // 10)  # ~10% of original
            else:
                effective_dims = min(
                    100, num_features // 20
                )  # More aggressive for very high-dim
        else:
            effective_dims = num_features

        min_samples = max(5, 2 * effective_dims)

        # min_cluster_size: scale-based heuristic
        if num_samples < 500:
            min_cluster_size = max(5, num_samples // 50)
        elif num_samples < 5000:
            min_cluster_size = max(15, num_samples // 100)
        elif num_samples < 50000:
            min_cluster_size = max(25, num_samples // 200)
        else:
            min_cluster_size = max(50, num_samples // 500)

        min_cluster_size = max(min_cluster_size, min_samples)

        metric = "manhattan" if effective_dims > 30 else "euclidean"

        hdbscan_clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            algorithm="auto",
            cluster_selection_method="eom",
            allow_single_cluster=False,
        )

        if not needs_reduction:
            return hdbscan_clusterer

        dimensionality_reduction = UMAP(
            n_neighbors=min(30, num_samples // 10),  # Cap at dataset size
            n_components=effective_dims,
            min_dist=0.0,  # Tight clusters for HDBSCAN
            metric="cosine",  # Often better for high-dim data
            random_state=self.random_state,  # Reproducibility
        )

        pipeline = Pipeline(
            [
                ("dimensionality_reduction", dimensionality_reduction),
                ("clustering", hdbscan_clusterer),
            ],
            memory=None,
        )

        return pipeline

    def _get_inductive_clusterer(self) -> tuple[object, object]:
        table_encoder = self.table_encoder or TransformToNumerical()
        classifier = self.classifier or KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )

        return clone(table_encoder), clone(classifier)

    def _determine_mode(self, cluster_labels: NDArray[np.integer]) -> str:
        """

        Args:
            cluster_labels:

        Returns:

        """
        if self.mode in ["classification", "regression"]:
            return self.mode

        # Auto mode: classification if ≤ max_clusters, else regression
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        return "classification" if n_clusters <= self.max_clusters else "regression"

    def _get_targets(
        self,
        X: NDArray,
        cluster_labels: NDArray[np.integer],
        mode: str,
    ) -> NDArray:
        """

        Args:
            X:
            cluster_labels:
            mode:

        Returns:

        """
        if mode == "classification":
            # Merge clusters if there are too many
            n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
            if n_clusters > self.max_clusters:
                return merge_clusters_to_max_k(X, cluster_labels, self.max_clusters)
            return cluster_labels
        else:  # regression mode
            # Get distance-based targets
            min_dist, mean_dist, std_dist = get_cluster_distance_targets(
                X, cluster_labels
            )

            targets = []
            if "min" in self.distance_metrics:
                targets.append(min_dist)
            if "mean" in self.distance_metrics:
                targets.append(mean_dist)
            if "std" in self.distance_metrics:
                targets.append(std_dist)

            # Stack into (n_samples, n_metrics) if multiple metrics
            if len(targets) == 1:
                return targets[0]
            return np.column_stack(targets)

    def _compute_embeddings(
        self,
        X_train_preprocessed: NDArray | pd.DataFrame,
        X_test_preprocessed: NDArray | pd.DataFrame | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> tuple[NDArray, NDArray | None]:
        """
        Computes embeddings for training and optional test datasets using an inductive clusterer.

        This method fits an inductive clusterer on the training data, generates cluster labels
        and corresponding targets, and computes embeddings. If test data is provided, it processes
        the test data through the trained clusterer and computes embeddings for the combined dataset.

        Args:
            X_train_preprocessed (NDArray | pd.DataFrame): Preprocessed training data used for
                clustering and embedding computation.
            X_test_preprocessed (NDArray | pd.DataFrame | None): Preprocessed test data for optional
                embedding computations. If None, embeddings are computed only for training data.
            outlier (bool): If True, handles outlier detection during cluster fitting and label
                generation.
            **kwargs: Additional arguments passed to underlying methods during computation.

        Returns:
            tuple[NDArray, NDArray | None]: A tuple containing:
                - Training data embeddings as a NumPy array.
                - Test data embeddings as a NumPy array, or None if test data is not provided.
        """

        # Setup and fit inductive clusterer
        table_encoder, classifier = self._get_inductive_clusterer()

        # Fit on training data
        X_train_encoded = table_encoder.fit_transform(X_train_preprocessed)
        num_samples, num_features = X_train_encoded.shape

        clusterer = self.clusterer or self.get_hdbscan_pipeline(
            num_samples=num_samples, num_features=num_features
        )

        train_cluster_labels = clusterer.fit_predict(X_train_encoded)

        # Store cluster count for reference
        self.n_clusters_ = len(
            np.unique(train_cluster_labels[train_cluster_labels != -1])
        )

        # Fit classifier for inductive predictions
        classifier.fit(X_train_encoded, train_cluster_labels)

        # Determine mode and generate targets
        self.actual_mode_ = self._determine_mode(train_cluster_labels)
        train_targets = self._get_targets(
            X_train_encoded, train_cluster_labels, self.actual_mode_
        )

        # Compute embeddings for training data
        X_train_embedded = self._compute_embeddings_internal(
            X_train_preprocessed, train_targets, self.actual_mode_
        )

        if X_test_preprocessed is None:
            return X_train_embedded, None

        # Predict cluster labels for test data
        X_test_encoded = table_encoder.transform(X_test_preprocessed)
        test_cluster_labels = classifier.predict(X_test_encoded)

        # Generate targets for test data
        test_targets = self._get_targets(
            X_test_encoded, test_cluster_labels, self.actual_mode_
        )

        # Stack train and test data
        X_train_test_stack = np.vstack([X_train_preprocessed, X_test_preprocessed])

        # Handle 1D vs 2D targets
        if len(train_targets.shape) == 1:
            targets_stack = np.hstack([train_targets, test_targets])
        else:
            targets_stack = np.vstack([train_targets, test_targets])

        # Compute embeddings for combined data
        X_train_test_embedded = self._compute_embeddings_internal(
            X_train_test_stack, targets_stack, self.actual_mode_
        )

        train_size = X_train_preprocessed.shape[0]
        return X_train_embedded, X_train_test_embedded[train_size:]

    def _compute_embeddings_internal(
        self,
        X: NDArray,
        targets: NDArray,
        mode: str,
    ) -> np.ndarray:
        """Internal method to compute embeddings from targets

        Args:
            X:
            targets:
            mode:
        """

        # Ensure targets is 2D
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)

        num_targets = targets.shape[1]

        # Select appropriate TabPFN model
        model = self.tabpfn_clf if mode == "classification" else self.tabpfn_reg

        target_embeddings = []

        for i in range(num_targets):
            target = targets[:, i]
            model.fit(X, target)

            tmp_embeddings = model.get_embeddings(X)

            # Handle multiple estimators
            if self.num_estimators > 1:
                if self.estimator_agg == "mean":
                    tmp_embeddings = np.mean(tmp_embeddings, axis=0)
                elif self.estimator_agg == "first_element":
                    tmp_embeddings = np.squeeze(tmp_embeddings[0, :])
                else:
                    raise NotImplementedError(
                        f"Aggregation '{self.estimator_agg}' not implemented"
                    )
            else:
                tmp_embeddings = np.squeeze(tmp_embeddings)

            target_embeddings.append(tmp_embeddings)

        # Concatenate embeddings from all targets
        concat_embeddings = np.concatenate(target_embeddings, axis=1).reshape(
            target_embeddings[0].shape[0], -1
        )

        # Apply embedding aggregation
        if self.emb_agg == "concat":
            return concat_embeddings
        elif self.emb_agg == "mean":
            reshaped_embeddings = concat_embeddings.reshape(
                *concat_embeddings.shape[:-1], num_targets, self.tabpfn_dim
            )
            embeddings = np.mean(reshaped_embeddings, axis=-2)
            return embeddings
        else:
            raise NotImplementedError(f"Aggregation '{self.emb_agg}' not implemented")

    def _reset_embedding_model(self):
        super()._reset_embedding_model()
        self.actual_mode_ = None
        self.n_clusters_ = None


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
            self.tabpfn_clf if task_type in CLASSIFICATION_TASKS else self.tabpfn_reg
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
