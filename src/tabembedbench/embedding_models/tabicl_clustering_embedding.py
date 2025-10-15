import inspect
from pathlib import Path
from typing import Tuple, Union

import warnings

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from skrub import TableVectorizer
from tabicl.sklearn.preprocessing import (
    PreprocessingPipeline,
    TransformToNumerical
)

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.embedding_models.tabicl_embedding import TabICLRowEmbedding
from tabembedbench.utils.torch_utils import get_device


class TabICLClusteringEmbedding(AbstractEmbeddingGenerator):
    """TabICL-based embedding generator for tabular data.

    This embedding model uses the TabICL (Tabular In-Context Learning) architecture
    to generate embeddings. It supports optional preprocessing pipelines and can
    download pre-trained models from HuggingFace Hub.

    Attributes:
        model_path (Path | None): Path to the TabICL model checkpoint.
        tabicl_row_embedder (TabICLRowEmbedding): The row embedding model loaded
            with pre-trained weights.
        normalize_embeddings (bool): Whether to normalize generated embeddings.
        preprocess_pipeline (PreprocessingPipeline): Pipeline for standard preprocessing.
        outlier_preprocessing_pipeline (OutlierPreprocessingPipeline): Pipeline for
            outlier detection preprocessing.
        _preprocess_tabicl_data (bool): Whether to apply TabICL-specific preprocessing.
        _tabvectorizer_preprocess (bool): Whether to apply TableVectorizer preprocessing.
        _tabvectorizer (TableVectorizer | None): TableVectorizer instance if enabled.

    References:
        Qu, J. et al. (2025). Tabicl: A tabular foundation model for in-context
        learning on large data. arXiv preprint arXiv:2502.05564.
    """

    def __init__(
        self,
        model_path: str | None = None,
        preprocess_tabicl_data: bool = False,
        num_samples_per_center: int = 50,
        random_state: int = 42,
    ):
        super().__init__(name="TabICLClusteringEmbedding")

        self.model_path = Path(model_path) if model_path is not None else None
        self.tabicl_row_embedder = self.get_tabicl_model()

        self.num_samples_per_center=num_samples_per_center
        self._preprocess_tabicl_data = preprocess_tabicl_data
        self.random_state = random_state

    def get_tabicl_model(self):
        """Load or download the TabICL model.

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
        self, X, train: bool = True, outlier:bool = False, **kwargs
    ) -> np.ndarray:
        """Preprocess input data using TabICL-specific pipelines.

        Applies optional TableVectorizer preprocessing followed by either standard
        or outlier-specific preprocessing pipelines based on the data type.

        Args:
            X (np.ndarray): Input data to preprocess.
            train (bool, optional): Whether to fit the preprocessing pipelines.
                Defaults to True.
            outlier (bool, optional): Whether to use outlier-specific preprocessing.
                Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Preprocessed data ready for embedding computation.
        """

        numerical_transformer = TransformToNumerical()
        preprocess_pipeline = PreprocessingPipeline()

        train_indices = kwargs.get("train_indices")
        test_indices = kwargs.get("test_indices")
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        X_train = numerical_transformer.fit_transform(X_train)
        X_test = numerical_transformer.transform(X_test)

        if self._preprocess_tabicl_data:
                X_train = preprocess_pipeline.fit_transform(X_train)
                X_test = preprocess_pipeline.transform(X_test)

        X_preprocessed = np.empty(X.values.shape, dtype=np.float64)
        X_preprocessed[train_indices] = X_train
        X_preprocessed[test_indices] = X_test

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
        self, X: np.ndarray, device: torch.device | None = None, **kwargs
    ) -> np.ndarray:
        """
        Computes embeddings for input data using a neural network-based row embedder.
        The method processes the input array into training and test sets to generate
        embeddings for each row.

        Args:
            X (np.ndarray): Input data for which embeddings are to be generated.
                Must be a 2D or 3D array.
            device (torch.device | None): The PyTorch device where computation
                will be performed. If not provided, the default device will be determined
                using `get_device`.
            **kwargs: Additional keyword arguments:
                - train_indices (sequence of int): Indices of rows in `X` used for training.
                - test_indices (sequence of int): Indices of rows in `X` used for testing.

        Returns:
            np.ndarray: Array of computed embeddings matching the shape of `X`. Each row
                contains the embedding corresponding to the input rows.
        """
        if device is None:
            device = get_device()
            self.tabicl_row_embedder.to(device)

        if len(X.shape) not in [2, 3]:
            raise ValueError("Input must be 2D or 3D array")

        train_indices = kwargs.get("train_indices")
        test_indices = kwargs.get("test_indices")
        X_train = X[train_indices]
        X_test = X[test_indices]

        X_train_torch = torch.from_numpy(X_train).float().to(device)

        if len(X_train_torch.shape) == 2:
            X_train_torch = X_train_torch.unsqueeze(0)

        embeddings_train = self.tabicl_row_embedder.forward(X_train_torch).cpu().squeeze().numpy()

        embeddings_train = embeddings_train.astype(np.float64)

        context_indices = self.get_clustering_center(embeddings_train)

        if len(X_train.shape) == 3:
            context_rows = X_train[0, context_indices]
        else:
            context_rows = X_train[context_indices]

        X_test_context = np.concat([X_test, context_rows], axis=0)

        X_test_context_torch = torch.from_numpy(X_test_context).float().to(device)

        if len(X_test_context_torch.shape) == 2:
            X_test_context_torch = X_test_context_torch.unsqueeze(0)

        embeddings_test_w_context = self.tabicl_row_embedder.forward(
            X_test_context_torch).cpu().squeeze().numpy()

        embeddings_test = embeddings_test_w_context[:len(test_indices)]

        return embeddings_train, embeddings_test

    def get_clustering_center(self, embeddings_train: np.ndarray) -> list:
        """
        Determines the closest embeddings to cluster centers using K-Means clustering.

        Groups the input embeddings into clusters, then identifies embeddings that are
        nearest to the centroids of these clusters. This method is widely used for
        selectively sampling data and improving efficiency in various machine learning
        tasks. Embeddings are grouped in a specified number of clusters, and from
        each cluster, a predefined number of nearest samples are selected.

        Args:
            embeddings_train (np.ndarray): The data embeddings to be clustered,
                represented as a 2D array where rows correspond to individual
                embeddings and columns represent the embedding dimensions.

        Returns:
            list: A list of indices corresponding to the embeddings closest to
            their respective cluster centers.
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
            distances = np.sqrt(np.sum(diff ** 2, axis=1))


            num_samples = min(len(cluster_indices), self.num_samples_per_center)
            closest_indices_in_cluster = np.argsort(distances)[:num_samples]

            original_indices = cluster_indices[closest_indices_in_cluster]

            closest_indices.extend(original_indices)

        return closest_indices

    def get_num_clusters(self, num_embeddings: int):
        """
        Determines the number of clusters based on the given number of embeddings.

        The function selects an appropriate number of clusters depending on the
        range in which the input 'num_embeddings' falls. It returns different
        values for predefined ranges, optimizing the number of clusters for
        different scales of embeddings.

        Args:
            num_embeddings (int): The total number of embeddings to consider
                for clustering. Must be a non-negative integer.

        Returns:
            int: The number of clusters determined based on the input
                num_embeddings.
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
        pass


def filter_params_for_class(cls, params_dict):
    """Filter parameters dictionary to only include valid parameters for a class.

    Args:
        cls: The class to filter parameters for.
        params_dict (dict): Dictionary of parameters to filter.

    Returns:
        dict: Filtered dictionary containing only valid parameters for the class.
    """
    sig = inspect.signature(cls.__init__)

    valid_params = set(sig.parameters.keys()) - {"self"}

    return {k: v for k, v in params_dict.items() if k in valid_params}
