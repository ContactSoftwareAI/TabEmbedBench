import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import (
    LocalOutlierFactor,
)

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.dataset_utils import (
    download_adbench_tabular_datasets,
)
from tabembedbench.utils.embedding_utils import check_nan
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device
from tabembedbench.utils.tracking_utils import update_result_dict

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    neighbors: int = 51,
    neighbors_step: int = 5,
    distance_metrics: list[str] = ["euclidean", "cosine"],
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
        neighbors: Integer specifying the number of neighbors to use for outlier
        neighbors_step: Integer specifying the step size for neighbors.
            Defaults to 5.
        distance_metrics: List of distance metrics to use for outlier detection.
            Defaults to ["euclidean", "cosine"].

    Returns:
        pl.DataFrame: A Polars DataFrame containing the benchmark results, including
            dataset names, dataset sizes, embedding model names, number of neighbors
            used for outlier detection, AUC scores, computation times for embeddings,
            and the benchmark category.

    References:
        [1] Han, S., et al. (2022). "Adbench: Anomaly detection benchmark."
            Advances in neural information processing systems, 35, 32142-32159.
    """
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

    result_outlier_dict = {
        "dataset_name": [],
        "dataset_size": [],
        "embedding_model": [],
        "num_neighbors": [],
        "auc_score": [],
        "time_to_compute_embeddings": [],
        "benchmark": [],
        "distance_metric": [],
        "task": []
    }

    for dataset_file in dataset_paths.glob("*.npz"):
        if dataset_file.name not in exclude_datasets:
            logger.info(f"Running benchmark for {dataset_file.name}...")

            if "dataset_name" not in result_outlier_dict:
                result_outlier_dict["dataset_name"] = []

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

                    for num_neighbors in range(1, neighbors, neighbors_step):
                        for distance_metric in distance_metrics:
                            score_auc, exception = (
                                _evaluate_local_outlier_factor(
                                num_neighbors=num_neighbors,
                                X_embed=X_embed,
                                y_true=y,
                                distance_metric=distance_metric,
                            ))

                            if exception is not None:
                                logger.warning(
                                    f"Error occurred while running experiment for "
                                    f"{embedding_model.name} with "
                                    f"Local Outlier Factor: {exception}"
                                )
                                continue

                            update_result_dict(
                                result_dict=result_outlier_dict,
                                dataset_name=dataset_file.stem,
                                dataset_size=X.shape[0],
                                embedding_model_name=embedding_model.name,
                                num_neighbors=num_neighbors,
                                compute_time=compute_embeddings_time,
                                task="Outlier Detection",
                                auc_score=score_auc,
                                distance_metric=distance_metric,
                                outlier_benchmark=True
                            )
                    logger.debug(
                        f"Finished experiment for {embedding_model.name} and "
                        f"resetting the model."
                    )
                    embedding_model.reset_embedding_model()

                if get_device() in ["cuda", "mps"]:
                    empty_gpu_cache()

    result_df = pl.from_dict(
        result_outlier_dict,
        schema={
            "dataset_name": pl.Categorical,
            "dataset_size": pl.UInt64,
            "embedding_model": pl.Categorical,
            "num_neighbors": pl.UInt64,
            "auc_score": pl.Float64,
            "time_to_compute_embeddings": pl.Float64,
            "benchmark": pl.Categorical,
            "distance_metric": pl.Categorical,
            "task": pl.Categorical,
        },
    )

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
