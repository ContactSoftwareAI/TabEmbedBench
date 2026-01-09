from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from skrub import TableVectorizer
from tabicl.sklearn.preprocessing import TransformToNumerical


def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )


def merge_clusters_to_max_k(
    X: NDArray, cluster_labels: NDArray[np.integer], max_clusters: int = 10
) -> NDArray[np.integer]:
    """
    Reduces the number of clusters in the dataset to a specified maximum limit
    by merging similar clusters using an agglomerative clustering approach.

    Args:
        X (NDArray): The dataset, represented as a NumPy array of shape
            (n_samples, n_features).
        cluster_labels (NDArray[np.integer]): An array of integer labels
            representing the current cluster assignments for each sample.
            Labels should match the corresponding samples in `X`.
        max_clusters (int): The maximum number of clusters to retain after
            merging. Default is 10.

    Returns:
        NDArray[np.integer]: An array of cluster labels after merging, with
        the total number of clusters reduced to the specified maximum.
    """
    unique_clusters = np.unique(cluster_labels)

    unique_clusters = unique_clusters[unique_clusters != -1]

    if len(unique_clusters) <= max_clusters:
        return cluster_labels

    centroids = np.array([X[cluster_labels == c].mean(axis=0) for c in unique_clusters])

    merger = AgglomerativeClustering(
        n_clusters=max_clusters,
        linkage="ward",
    )

    meta_clusters = merger.fit_predict(centroids)

    cluster_mapping = dict(zip(unique_clusters, meta_clusters))

    merged_labels = np.array(
        [cluster_mapping.get(label, -1) for label in cluster_labels]
    )

    return merged_labels


def get_cluster_distance_targets(
    X: NDArray, cluster_labels: NDArray[np.integer]
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate cluster distance metrics for each sample.

    This function computes three different distance metrics for each sample in the dataset
    relative to identified clusters. The metrics include the minimum distance to a cluster
    centroid, the weighted mean distance to cluster centroids, and the standard deviation
    of distances from the sample to all cluster centroids.

    Args:
        X (NDArray): An array of shape (n_samples, n_features) representing the dataset.
        cluster_labels (NDArray[np.integer]): An array of shape (n_samples,) where each entry
            indicates the cluster assignment for the corresponding sample.
            A value of -1 indicates that the sample does not belong to any cluster.

    Returns:
        Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: The tuple
        containing three arrays:
            - min_distances (NDArray[np.floating]): A 1-dimensional array of shape
              (n_samples,) representing the negative minimum Euclidean distance for each
              sample to the nearest cluster centroid.
            - mean_distances (NDArray[np.floating]): A 1-dimensional array of shape
              (n_samples,) representing the negative weighted mean Euclidean distance for
              each sample to all cluster centroids.
            - std_distances (NDArray[np.floating]): A 1-dimensional array of shape
              (n_samples,) representing the standard deviation of distances from each
              sample to all cluster centroids.
    """
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])

    centroids = np.array([X[cluster_labels == c].mean(axis=0) for c in unique_clusters])
    cluster_sizes = np.array([(cluster_labels == c).sum() for c in unique_clusters])

    weights = cluster_sizes / cluster_sizes.sum()

    distances = pairwise_distances(X, centroids, metric="euclidean")

    min_distances = (-1) * distances.min(axis=1)
    mean_distances = (-1) * (distances * weights).sum(axis=1)
    std_distances = distances.std(axis=1)

    return min_distances, mean_distances, std_distances
