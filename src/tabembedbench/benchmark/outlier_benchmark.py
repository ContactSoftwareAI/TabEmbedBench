import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import (
    LocalOutlierFactor,
)

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.dataset_utils import (
    download_adbench_tabular_datasets
)
from tabembedbench.utils.embedding_utils import check_nan
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device
from tabembedbench.utils.tracking_utils import (
    get_batch_dict_result_df,
    update_batch_dict,
    update_result_df,
    save_result_df
)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = get_benchmark_logger("TabEmbedBench_Outlier")

IMAGE_CATEGORY = [
    "1_ALOI.npz",
    "8_celeba.npz",
    "17_InternetAds.npz",
    "20_letter.npz",
    "24_mnist.npz",
    "26_optdigits.npz",
    "28_pendigits.npz",
    "33_skin.npz",
]


def run_outlier_benchmark(
    embedding_models: list[BaseEmbeddingGenerator],
    dataset_paths: str | Path | None = None,
    exclude_datasets: list[str] | None = None,
    exclude_image_datasets: bool = False,
    save_embeddings: bool = False,
    upper_bound_num_samples: int = 10000,
    upper_bound_num_features: int = 500,
    neighbors: int = 51,
    neighbors_step: int = 5,
    distance_metrics=None,
    save_result_dataframe: bool = True,
    result_dir: str | Path = "result_outlier",
    timestamp: str = TIMESTAMP,
):
    """Runs an outlier detection benchmark using the provided embedding models
    and datasets. It uses the tabular datasets from the ADBench benchmark [1]
    for evaluation.

    This function benchmarks the effectiveness of various embedding models in
    detecting outliers. It supports the exclusion of specific datasets,
    exclusion of image datasets, limiting the dataset size, and optionally
    saving computed embeddings for analysis.

    Args:
        embedding_models: A list of embedding models to be evaluated. Each
            embedding model must implement methods for preprocessing data,
            computing embeddings, and resetting the model.
        dataset_paths: Optional path to the dataset directory. If not specified,
            a default directory for tabular datasets will be used,
            and datasets will be downloaded if missing.
        exclude_datasets: Optional list of dataset filenames to exclude from the
            benchmark. Each filename should match a file in the dataset directory.
        exclude_image_datasets: Boolean flag that indicates whether to exclude
            image datasets from the benchmark. Defaults to False.
        save_embeddings: Boolean flag to determine whether computed embeddings
            should be saved to disk. Defaults to False.
        upper_bound_num_samples: Integer specifying the maximum size of rows
            (in number of samples) to include in the benchmark. Datasets exceeding
            this size will be skipped. Defaults to 10000.
        upper_bound_num_features: Integer specifying the maximum number of features
            to include in the benchmark. Datasets with more features than this
            value will be skipped. Defaults to 500.
        neighbors: Integer specifying the number of neighbors to use for outlier
        neighbors_step: Integer specifying the step size for neighbors.
            Defaults to 5.
        distance_metrics: List of distance metrics to use for outlier detection.
            Defaults to ["euclidean", "cosine"].
        save_result_dataframe: Boolean flag to determine whether to save the result
            dataframe to disk. Defaults to True.
        result_dir: Optional path to the directory where the result dataframe should
            be saved. Defaults to "result_outlier".
        timestamp: Optional timestamp string to use for saving the result dataframe.
            Defaults to the current timestamp.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the benchmark results, including
            dataset names, dataset sizes, embedding model names, number of neighbors
            used for outlier detection, AUC scores, computation times for embeddings,
            and the benchmark category.

    References:
        [1] Han, S., et al. (2022). "Adbench: Anomaly detection benchmark."
            Advances in neural information processing systems, 35, 32142-32159.
    """
    if distance_metrics is None:
        distance_metrics = ["euclidean", "cosine"]
    if dataset_paths is None:
        dataset_paths = Path("data/adbench_tabular_datasets")
        if not dataset_paths.exists():
            logger.warning("Downloading ADBench tabular datasets...")
            download_adbench_tabular_datasets(dataset_paths)
    else:
        dataset_paths = Path(dataset_paths)

    if exclude_image_datasets:
        if exclude_datasets is not None:
            exclude_datasets.extend(IMAGE_CATEGORY)
        else:
            exclude_datasets = IMAGE_CATEGORY

    if isinstance(result_dir, str):
        result_dir = Path(result_dir)

    batch_dict, result_df = get_batch_dict_result_df()

    for dataset_file in dataset_paths.glob("*.npz"):
        if dataset_file.name not in exclude_datasets:
            logger.info(f"Running benchmark for {dataset_file.name}...")

            if "dataset_name" not in batch_dict:
                batch_dict["dataset_name"] = []

            with np.load(dataset_file) as dataset:
                num_samples = dataset["X"].shape[0]
                dataset_name = dataset_file.stem
                num_features = dataset["X"].shape[1]

                if num_samples > upper_bound_num_samples:
                    logger.warning(
                        f"Skipping {dataset_name} "
                        f"- dataset size {num_samples}"
                        f"exceeds limit {upper_bound_num_samples}"
                    )
                    continue
                if num_features > upper_bound_num_features:
                    logger.warning(
                        f"Skipping {dataset_name} "
                        f"- number of features size {num_features}"
                        f"exceeds limit {upper_bound_num_features}"
                    )
                    continue

                logger.info(
                    f"Running experiments on {dataset_name}. "
                    f"Samples: {num_samples}, "
                    f"Features: {num_features}"
                )

            dataset = np.load(dataset_file)

            X = dataset["X"]
            y = dataset["y"]

            for embedding_model in embedding_models:
                logger.debug(f"Starting experiment for "
                             f"{embedding_model.name}..."
                             f"Compute Embeddings.")

                X_embed, compute_embeddings_time = embedding_model.compute_embeddings(X)

                if save_embeddings:
                    embedding_file = (
                        f"{embedding_model.name}"
                        f"_{dataset_file.stem}_embeddings.npz"
                    )
                    np.savez(embedding_file, x=X_embed, y=y)

                    os.remove(embedding_file)

                if check_nan(X_embed):
                    logger.warning(
                        f"The embeddings for {dataset_file.name} contain NaN "
                        f"values with embedding model {embedding_model.name}. "
                        f"Skipping."
                    )
                else:
                    logger.debug(
                        f"Start Outlier Detection for {embedding_model.name} "
                        f"with Local Outlier Factor."
                    )

                    for num_neighbors in range(0, neighbors, neighbors_step):
                        if num_neighbors == 0:
                            continue
                        for distance_metric in distance_metrics:
                            score_auc, exception_message = (
                                _evaluate_local_outlier_factor(
                                num_neighbors=num_neighbors,
                                X_embed=X_embed,
                                y_true=y,
                                distance_metric=distance_metric,
                            ))

                            if exception_message is not None:
                                logger.warning(
                                    f"Error occurred while running experiment for "
                                    f"{embedding_model.name} with "
                                    f"Local Outlier Factor: {exception_message}"
                                )
                                continue

                            update_batch_dict(
                                batch_dict=batch_dict,
                                dataset_name=dataset_file.stem,
                                dataset_size=X.shape[0],
                                embedding_model_name=embedding_model.name,
                                num_neighbors=num_neighbors,
                                compute_time=compute_embeddings_time,
                                task="Outlier Detection",
                                auc_score=score_auc,
                                distance_metric=distance_metric,
                                algorithm="LocalOutlierFactor"
                            )

                    # Isolation Forest Implementation
                    logger.debug(
                        f"Start Outlier Detection for {embedding_model.name} "
                        f"with Isolation Forest."
                    )

                    score_auc, exception_message = _evaluate_isolation_forest(
                        X_embed=X_embed,
                        y_true=y,
                    )

                    if exception_message is not None:
                        logger.warning(
                            f"Error occurred while running experiment for "
                            f"{embedding_model.name} with "
                            f"Isolation Forest: {exception_message}"
                        )
                    else:
                        update_batch_dict(
                            batch_dict=batch_dict,
                            dataset_name=dataset_file.stem,
                            dataset_size=X.shape[0],
                            embedding_model_name=embedding_model.name,
                            num_neighbors=0,  # Not applicable for Isolation Forest
                            compute_time=compute_embeddings_time,
                            task="Outlier Detection",
                            auc_score=score_auc,
                            distance_metric='',  # Not applicable for Isolation Forest
                            algorithm="IsolationForest"
                        )

                logger.debug(
                        f"Finished experiment for {embedding_model.name} and "
                        f"resetting the model."
                    )
                embedding_model.reset_embedding_model()

                batch_dict, result_df = update_result_df(
                    batch_dict=batch_dict, result_df=result_df, logger=logger
                )

                if save_result_dataframe:
                    save_result_df(result_df=result_df,
                                   output_path=result_dir,
                                   benchmark_name="ADBench_Tabular",
                                   timestamp=timestamp)

                if get_device() in ["cuda", "mps"]:
                    empty_gpu_cache()

    return result_df


def _evaluate_local_outlier_factor(
    num_neighbors: int,
    X_embed: np.ndarray,
    y_true: np.ndarray,
    distance_metric: str = "euclidean",
):
    """
    Evaluates the Local Outlier Factor (LOF) model to identify outlier detection
    performance by computing the Area Under the Receiver Operating Characteristic
    (AUROC) score for given input data.

    This function utilizes the Local Outlier Factor algorithm to compute the
    outlier factor for each data point in the provided dataset. It then evaluates
    the performance of the outlier detection by calculating the ROC AUC score
    against the ground-truth labels.

    Args:
        num_neighbors (int): The number of neighbors to use for LOF calculation.
        X_embed (np.ndarray): The embedded feature space data used for LOF
            computation.
        y_true (np.ndarray): Ground-truth binary labels indicating which data
            points are outliers.
        distance_metric (str): The distance metric to use for nearest neighbor
            computation. Defaults to "euclidean".

    Returns:
        tuple: A tuple consisting of the following:
            - float: The computed ROC AUC score indicating model performance, or
              None in case of an error.
            - Exception: An exception object if an error occurred during calculation,
              otherwise None.
    """
    try:
        lof = LocalOutlierFactor(
            n_neighbors=num_neighbors,
            n_jobs=-1,
            metric=distance_metric,
        )

        lof.fit_predict(X_embed)
        neg_outlier_factor = (-1) * lof.negative_outlier_factor_
        score_auc = roc_auc_score(y_true, neg_outlier_factor)

        return score_auc, None
    except Exception as e:
        return None, e


def _evaluate_isolation_forest(
        X_embed: np.ndarray,
        y_true: np.ndarray,
        contamination: float = "auto",
        n_estimators: int = 100,
):
    """
    Evaluates the Isolation Forest model to identify outlier detection
    performance by computing the AUROC score for given input data.

    This function utilizes the Isolation Forest algorithm to compute the
    anomaly score for each data point in the provided dataset. It then evaluates
    the performance of the outlier detection by calculating the ROC AUC score
    against the ground-truth labels.

    Args:
        X_embed (np.ndarray): The embedded feature space data used for Isolation
            Forest computation.
        y_true (np.ndarray): Ground-truth binary labels indicating which data
            points are outliers.
        contamination (float or str): The amount of contamination of the data set,
            i.e., the proportion of outliers in the data set. Defaults to "auto".
        n_estimators (int): The number of base estimators in the ensemble.
            Defaults to 100.

    Returns:
        tuple: A tuple consisting of the following:
            - float: The computed ROC AUC score indicating model performance, or
              None in case of an error.
            - Exception: An exception object if an error occurred during calculation,
              otherwise None.
    """
    try:
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            n_jobs=-1,
        )

        iso_forest.fit(X_embed)
        # Get anomaly scores (negative values for outliers)
        anomaly_scores = iso_forest.decision_function(X_embed)
        # Convert to positive scores (higher values = more anomalous)
        outlier_scores = -anomaly_scores

        score_auc = roc_auc_score(y_true, outlier_scores)

        return score_auc, None
    except Exception as e:
        return None, e
