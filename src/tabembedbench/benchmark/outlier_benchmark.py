from pathlib import Path
import logging
import os
import time
from typing import List, Optional, Union

import mlflow
import numpy as np
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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

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
    upper_bound_dataset_size: int = 100000,
):
    if dataset_paths is None:
        dataset_paths = Path("data/adbench_tabular_datasets")
        if not dataset_paths.exists():
            logger.warning("Downloading ADBench tabular datasets...")
            download_adbench_tabular_datasets(dataset_paths)

    if exclude_image_datasets:
        if exclude_datasets is not None:
            exclude_datasets.extend(IMAGE_CATEGORY)
        else:
            exclude_datasets = IMAGE_CATEGORY

    for dataset_file in dataset_paths.glob("*.npz"):
        if dataset_file.name not in exclude_datasets:
            logger.info(f"Running benchmark for {dataset_file.name}...")

            with np.load(dataset_file) as dataset:
                if dataset["X"].shape[0] > upper_bound_dataset_size:
                    logger.warning(
                        f"Skipping {dataset_file.name} - dataset size {dataset['X'].shape[0]} exceeds limit {upper_bound_dataset_size}"
                    )
                    continue

            dataset = np.load(dataset_file)

            X = dataset["X"]
            y = dataset["y"]

            for embedding_model in embedding_models:
                with mlflow.start_run(
                    run_name=f"outlier_{embedding_model.name}_{dataset_file.stem}",
                    nested=True,
                ):
                    X = embedding_model.preprocess_data(X, train=True)

                    start_time = time.time()
                    X_embed = embedding_model.compute_embeddings(X)
                    compute_embeddings_time = time.time() - start_time

                    if save_embeddings:
                        embedding_file = (
                            f"{embedding_model.name}_{dataset_file.stem}_embeddings.npz"
                        )
                        np.savez(embedding_file, x=X_embed, y=y)

                        mlflow.log_artifact(embedding_file)

                        os.remove(embedding_file)

                    mlflow.log_metric("compute_embedding_time", compute_embeddings_time)

                    for num_neighbors in range(1, 51):
                        lof = LocalOutlierFactor(
                            n_neighbors=num_neighbors,
                            n_jobs=-1,
                        )

                        X = lof.fit_predict(X_embed)

                        neg_outlier_factor = (-1) * lof.negative_outlier_factor_

                        score_auc = roc_auc_score(y, neg_outlier_factor)

                        mlflow.log_metric(
                            "auc_score",
                            score_auc,
                            step=num_neighbors,
                        )
