from pathlib import Path
from typing import Tuple

import warnings

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from tabicl.sklearn.preprocessing import PreprocessingPipeline

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.embedding_models.tabicl_embedding import (
    TabICLRowEmbedding,
    filter_params_for_class,
)
from tabembedbench.utils.torch_utils import get_device


class TabICLClusteringEmbedding(AbstractEmbeddingGenerator):
    """TabICL-based embedding generator with clustering-based context selection.

    This class extends the standard TabICL embedding approach by using K-means
    clustering to select representative training samples as context for test data
    embeddings. This leverages TabICL's in-context learning capabilities by providing
    diverse, representative examples from the training set.

    The embedding generation process:
    1. Generate embeddings for all training data using TabICL
    2. Cluster training embeddings using K-means
    3. Select samples closest to each cluster centroid as context examples
    4. Concatenate context examples with test data
    5. Generate final test embeddings with context

    This approach can improve embedding quality by providing relevant context that
    captures the diversity of the training distribution.

    Attributes:
        model_path (Path | None): Path to a local TabICL model checkpoint.
        tabicl_row_embedder (TabICLRowEmbedding): The loaded TabICL model with frozen
            parameters.
        num_samples_per_center (int): Number of samples to select per cluster centroid
            as context examples.
        preprocess_pipeline (PreprocessingPipeline | None): Fitted preprocessing pipeline.
        random_state (int): Random seed for K-means clustering.
        device (str): Device for computation ('cuda' or 'cpu').

    Example:
        >>> embedding_gen = TabICLClusteringEmbedding(num_samples_per_center=50)
        >>> train_emb, test_emb, time = embedding_gen.generate_embeddings(
        ...     X_train, X_test
        ... )
    """

    def __init__(
        self,
        model_path: str | None = None,
        num_samples_per_center: int = 50,
        random_state: int = 42,
        device: str | None = None,
    ):
        """Initialize the TabICL clustering-based embedding generator.

        Args:
            model_path (str | None, optional): Path to a local TabICL model checkpoint.
                If None, downloads from HuggingFace. Defaults to None.
            num_samples_per_center (int, optional): Number of samples to select per
                cluster centroid as context examples. Defaults to 50.
            random_state (int, optional): Random seed for K-means clustering to ensure
                reproducibility. Defaults to 42.
            device (str | None, optional): Device to use for computation ('cuda' or 'cpu').
                If None, automatically detects GPU availability. Defaults to None.
        """
        super().__init__(name="TabICLClusteringEmbedding")

        self.model_path = Path(model_path) if model_path is not None else None
        self.tabicl_row_embedder = self.get_tabicl_model()

        self.num_samples_per_center = num_samples_per_center
        self.preprocess_pipeline = None
        self.random_state = random_state
        self.device = device if device is not None else get_device()

    def get_tabicl_model(self):
        """Load or download the TabICL model with pre-trained weights.

        This method either loads a model from a local checkpoint or downloads
        the pre-trained TabICL classifier from HuggingFace Hub. The model's
        parameters are frozen to prevent updates during embedding generation.

        Returns:
            TabICLRowEmbedding: The loaded TabICL row embedding model with
                pre-trained weights and frozen parameters.
        """
        if self.model_path is not None and self.model_path.exists():
            model_ckpt_path = self.model_path
        else:
            model_ckpt_path = hf_hub_download(
                repo_id="jingang/TabICL-clf",
                filename="tabicl-classifier-v1.1-0506.ckpt",
            )

        model_ckpt = torch.load(model_ckpt_path)

        state_dict = model_ckpt["state_dict"]
        config = model_ckpt["config"]

        filtered_config = filter_params_for_class(TabICLRowEmbedding, config)

        row_embedding_model = TabICLRowEmbedding(
            **filtered_config,
        )

        row_embedding_model.load_state_dict(state_dict, strict=False)

        for param in row_embedding_model.col_embedder.parameters():
            param.requires_grad = False

        for param in row_embedding_model.row_interactor.parameters():
            param.requires_grad = False

        row_embedding_model.eval()

        return row_embedding_model

    def _preprocess_data(
        self, X: np.ndarray, train: bool = True, outlier: bool = False, **kwargs
    ) -> np.ndarray:
        """Preprocess input data using TabICL-specific preprocessing pipeline.

        Applies TabICL's standard preprocessing which includes scaling and
        normalization to prepare data for the model.

        Args:
            X (np.ndarray): Input data to preprocess.
            train (bool, optional): Whether to fit the preprocessing pipeline.
                Defaults to True.
            outlier (bool, optional): Whether to use outlier-specific preprocessing
                (currently unused in this class). Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Preprocessed data ready for embedding computation.

        Raises:
            ValueError: If preprocessing pipeline is not fitted when train is False.
        """
        if train:
            self.preprocess_pipeline = PreprocessingPipeline()
            X_preprocessed = self.preprocess_pipeline.fit_transform(X)
        else:
            if self.preprocess_pipeline is None:
                raise ValueError("Preprocessing pipeline is not fitted")
            else:
                X_preprocessed = self.preprocess_pipeline.transform(X)

        return X_preprocessed

    def _fit_model(self, X_preprocessed: np.ndarray, train: bool = True, **kwargs):
        """Fit the model (no-op for TabICL as it uses pre-trained weights).

        Args:
            X_preprocessed (np.ndarray): Preprocessed input data.
            train (bool, optional): Whether this is training mode. Defaults to True.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: The preprocessed data unchanged.
        """
        return X_preprocessed

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray,
        X_test_preprocessed: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Compute embeddings using TabICL with clustering-based context selection.

        This method generates embeddings for training data, performs K-means clustering
        to identify representative samples, and uses these as context when generating
        test embeddings. This leverages TabICL's in-context learning capabilities.

        The process:
        1. Generate initial embeddings for all training data
        2. Cluster training embeddings using K-means
        3. Select samples nearest to each cluster centroid
        4. Concatenate selected context samples with test data
        5. Generate final test embeddings with context

        Args:
            X_train_preprocessed (np.ndarray): Preprocessed training dataset. Must be
                2D or 3D array of shape (n_samples, n_features) or (1, n_samples, n_features).
            X_test_preprocessed (np.ndarray | None, optional): Preprocessed test dataset.
                Must be a 2D or 3D array. Required for standard mode. Defaults to None.
            outlier (bool, optional): Whether this is outlier mode (currently not
                implemented for this class). Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            tuple[np.ndarray, np.ndarray | None]: A tuple containing:
                - train_embeddings: Embeddings for training data of shape (n_train, embed_dim)
                - test_embeddings: Embeddings for test data with context, shape (n_test, embed_dim)

        Raises:
            ValueError: If input data is not a 2D or 3D array.
        """
        self.tabicl_row_embedder.to(self.device)

        if len(X_train_preprocessed.shape) not in [2, 3]:
            raise ValueError("Input must be 2D or 3D array")

        X_train_torch = torch.from_numpy(X_train_preprocessed).float().to(self.device)

        if len(X_train_torch.shape) == 2:
            X_train_torch = X_train_torch.unsqueeze(0)

        embeddings_train = (
            self.tabicl_row_embedder.forward(X_train_torch).cpu().squeeze().numpy()
        )

        embeddings_train = embeddings_train.astype(np.float64)

        context_indices = self.get_clustering_center(embeddings_train)

        if len(X_train_preprocessed.shape) == 3:
            context_rows = X_train_preprocessed[0, context_indices]
        else:
            context_rows = X_train_preprocessed[context_indices]

        X_test_context = np.concat([context_rows, X_test_preprocessed], axis=0)

        X_test_context_torch = torch.from_numpy(X_test_context).float().to(self.device)

        if len(X_test_context_torch.shape) == 2:
            X_test_context_torch = X_test_context_torch.unsqueeze(0)

        embeddings_test_w_context = (
            self.tabicl_row_embedder.forward(X_test_context_torch)
            .cpu()
            .squeeze()
            .numpy()
        )

        embeddings_test = embeddings_test_w_context[len(context_indices) :]

        return embeddings_train, embeddings_test

    def get_clustering_center(self, embeddings_train: np.ndarray) -> list:
        """Identify representative training samples using K-means clustering.

        This method clusters the training embeddings and selects samples closest
        to each cluster centroid. These samples serve as diverse, representative
        context examples for generating test embeddings.

        Args:
            embeddings_train (np.ndarray): Training data embeddings of shape
                (n_samples, embed_dim).

        Returns:
            list: Indices of selected training samples that are closest to cluster
                centroids. The number of indices equals num_clusters * num_samples_per_center,
                though it may be less if some clusters are empty.

        Note:
            The number of clusters is automatically determined based on the number
            of training samples using the get_num_clusters static method.
        """
        num_embeddings = embeddings_train.shape[0]

        num_clusters = self.get_num_clusters(num_embeddings)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )

            kmeans.fit(embeddings_train)

        cluster_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        closest_indices = []

        for cluster_id in range(num_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                print(f"Warning: Cluster {cluster_id} is empty!")
                continue

            cluster_embeddings = embeddings_train[cluster_indices]

            centroid = centroids[cluster_id].reshape(1, -1)

            diff = cluster_embeddings - centroid
            distances = np.sqrt(np.sum(diff**2, axis=1))

            num_samples = min(len(cluster_indices), self.num_samples_per_center)
            closest_indices_in_cluster = np.argsort(distances)[:num_samples]

            original_indices = cluster_indices[closest_indices_in_cluster]

            closest_indices.extend(original_indices)

        return closest_indices

    @staticmethod
    def get_num_clusters(num_embeddings: int) -> int:
        """Determine the optimal number of clusters based on dataset size.

        This method provides a heuristic for selecting the number of K-means
        clusters based on the training set size. Larger datasets use more
        clusters to capture greater diversity.

        Args:
            num_embeddings (int): The total number of training samples.

        Returns:
            int: The number of clusters to use:
                - 20 for datasets < 1,000 samples
                - 100 for datasets < 10,000 samples
                - 200 for datasets < 100,000 samples
                - 500 for datasets >= 100,000 samples
        """
        if num_embeddings < 1000:
            return 20
        elif num_embeddings < 10000:
            return 100
        elif num_embeddings < 100000:
            return 200
        else:
            return 500

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Reinitializes all preprocessing pipelines to clear fitted state.
        """
        self.preprocess_pipeline = None
