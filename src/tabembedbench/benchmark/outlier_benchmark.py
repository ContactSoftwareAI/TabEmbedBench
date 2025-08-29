from pathlib import Path
import logging
import os
import time
from typing import List, Optional, Union

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

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger("TabEmbedBench_Outlier")

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
    embedding_models: List[BaseEmbeddingGenerator],
    dataset_paths: Optional[Union[str, Path]] = None,
    exclude_datasets: Optional[list[str]] = None,
    exclude_image_datasets: bool = False,
    save_embeddings: bool = False,
    upper_bound_dataset_size: int = 10000,
):
    """
    Runs an outlier detection benchmark using the provided embedding models and datasets.
    It uses the tabular datasets from the ADBench benchmark [1] for evaluation.

    This function benchmarks the effectiveness of various embedding models in detecting
    outliers. It supports the exclusion of specific datasets, exclusion of image datasets,
    limiting the dataset size, and optionally saving computed embeddings for analysis.

    Args:
        embedding_models: A list of embedding models to be evaluated. Each embedding
            model must implement methods for preprocessing data, computing embeddings,
            and resetting the model.
        dataset_paths: Optional path to the dataset directory. If not specified, a
            default directory for tabular datasets will be used, and datasets will
            be downloaded if missing.
        exclude_datasets: Optional list of dataset filenames to exclude from the
            benchmark. Each filename should match a file in the dataset directory.
        exclude_image_datasets: Boolean flag that indicates whether to exclude image
            datasets from the benchmark. Defaults to False.
        save_embeddings: Boolean flag to determine whether computed embeddings
            should be saved to disk. Defaults to False.
        upper_bound_dataset_size: Integer specifying the maximum dataset size
            (in number of samples) to include in the benchmark. Datasets exceeding
            this size will be skipped. Defaults to 10000.

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
    }

    for dataset_file in dataset_paths.glob("*.npz"):
        if dataset_file.name not in exclude_datasets:
            logger.info(f"Running benchmark for {dataset_file.name}...")

            if not "dataset_name" in result_outlier_dict.keys():
                result_outlier_dict["dataset_name"] = []

            with np.load(dataset_file) as dataset:
                if dataset["X"].shape[0] > upper_bound_dataset_size:
                    logger.warning(
                        f"Skipping {dataset_file.name} - dataset size {dataset['X'].shape[0]} exceeds limit {upper_bound_dataset_size}"
                    )
                    continue
                else:
                    logger.info(
                        f"Running experiments on {dataset_file.stem}. Samples: {dataset['X'].shape[0]}, Features: {dataset['X'].shape[1]}"
                    )

            dataset = np.load(dataset_file)

            X = dataset["X"]
            y = dataset["y"]

            for embedding_model in embedding_models:
                X_preprocess = embedding_model.preprocess_data(X, train=True)

                start_time = time.time()
                X_embed = embedding_model.compute_embeddings(X_preprocess)
                compute_embeddings_time = time.time() - start_time
                if save_embeddings:
                    embedding_file = (
                        f"{embedding_model.name}_{dataset_file.stem}_embeddings.npz"
                    )
                    np.savez(embedding_file, x=X_embed, y=y)

                    os.remove(embedding_file)

                for num_neighbors in range(1, 51):
                    lof = LocalOutlierFactor(
                        n_neighbors=num_neighbors,
                        n_jobs=-1,
                    )

                    lof.fit_predict(X_embed)

                    neg_outlier_factor = (-1) * lof.negative_outlier_factor_

                    score_auc = roc_auc_score(y, neg_outlier_factor)

                    result_outlier_dict["dataset_name"].append(dataset_file.stem)
                    result_outlier_dict["dataset_size"].append(X.shape[0])
                    result_outlier_dict["embedding_model"].append(embedding_model.name)
                    result_outlier_dict["num_neighbors"].append(num_neighbors)
                    result_outlier_dict["auc_score"].append(score_auc)
                    result_outlier_dict["time_to_compute_embeddings"].append(
                        compute_embeddings_time
                    )
                    result_outlier_dict["benchmark"].append("outlier")

                embedding_model.reset_embedding_model()

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
        },
    )

    return result_df
