from pathlib import Path
from typing import Tuple

import warnings

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from tabicl.sklearn.preprocessing import (
    PreprocessingPipeline
)

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.embedding_models.tabicl_embedding import (
    TabICLRowEmbedding,
    filter_params_for_class
)
from tabembedbench.utils.torch_utils import get_device


class TabICLClusteringEmbedding(AbstractEmbeddingGenerator):
    def __init__(
        self,
        model_path: str | None = None,
        num_samples_per_center: int = 50,
        random_state: int = 42,
        device: str | None = None,
    ):
        super().__init__(name="TabICLClusteringEmbedding")

        self.model_path = Path(model_path) if model_path is not None else None
        self.tabicl_row_embedder = self.get_tabicl_model()

        self.num_samples_per_center=num_samples_per_center
        self.preprocess_pipeline = None
        self.random_state = random_state
        self.device = device if device is not None else get_device()

    def get_tabicl_model(self):
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
        self,
        X: np.ndarray,
        train: bool = True,
        outlier: bool = False,
        **kwargs
    ) -> np.ndarray:
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
        return X_preprocessed

    def _compute_embeddings(
        self,
        X_train_preprocessed: np.ndarray,
        X_test_preprocessed: np.ndarray | None = None,
        outlier: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray | None]:
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

        X_test_context_torch = torch.from_numpy(X_test_context).float().to(
            self.device
        )

        if len(X_test_context_torch.shape) == 2:
            X_test_context_torch = X_test_context_torch.unsqueeze(0)

        embeddings_test_w_context = self.tabicl_row_embedder.forward(
            X_test_context_torch
        ).cpu().squeeze().numpy()

        embeddings_test = embeddings_test_w_context[len(context_indices):]

        return embeddings_train, embeddings_test

    def get_clustering_center(self, embeddings_train: np.ndarray) -> list:
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

    @staticmethod
    def get_num_clusters(num_embeddings: int):
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
        self.preprocess_pipeline = None
