import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.utils import infer_categorical_features

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.torch_utils import get_device


class UniversalTabPFNEmbedding(AbstractEmbeddingGenerator):
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
        return torch.from_numpy(X).float().to(self.device)

    def _fit_model(
        self,
        X_preprocessed: torch.Tensor,
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
            self.categorical_indices = infer_categorical_features(X_preprocessed)

        self.num_features = X_preprocessed.shape[-1]

        self._is_fitted = True

    def _compute_embeddings(
        self,
        X_preprocessed: torch.Tensor,
        **kwargs,
    ) -> np.ndarray:
        """Compute embeddings using TabPFN models.

        For each feature column, this method:
        1. Masks the feature as the target variable
        2. Uses remaining features as inputs to train a TabPFN model
        3. Extracts embeddings from the trained model
        4. Aggregates embeddings according to the specified strategy

        The method automatically selects TabPFNClassifier for categorical features
        and TabPFNRegressor for continuous features.

        Args:
            X_preprocessed (torch.Tensor): Preprocessed input data of shape
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
        num_samples = X_preprocessed.shape[0]
        tmp_embeddings = []

        for column_idx in range(X_preprocessed.shape[1]):
            mask = torch.zeros_like(X_preprocessed).bool()
            mask[:, column_idx] = True
            features, target = (
                X_preprocessed[~mask].reshape(num_samples, -1),
                X_preprocessed[mask],
            )

            model = (
                self.tabpfn_clf
                if column_idx in self.categorical_indices
                else self.tabpfn_reg
            )

            model.fit(features, target)
            estimator_embeddings = model.get_embeddings(features)

            if self.num_estimators > 1:
                if self.estimator_agg == "mean":
                    estimator_embeddings = np.mean(estimator_embeddings, axis=0)
                elif self.estimator_agg == "first_element":
                    estimator_embeddings = estimator_embeddings[0, :]
                else:
                    raise NotImplementedError

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
