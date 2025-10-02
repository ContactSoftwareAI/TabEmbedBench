from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.embedding_models import (
    AbstractEmbeddingGenerator
)
from tabembedbench.utils.dataset_utils import (
    download_adbench_tabular_datasets
)
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device
from tabembedbench.utils.tracking_utils import (
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
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    dataset_paths: str | Path | None = None,
    exclude_datasets: list[str] | None = None,
    exclude_image_datasets: bool = True,
    upper_bound_num_samples: int = 10000,
    upper_bound_num_features: int = 500,
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
        evaluators: A list of algorithm.
        dataset_paths: Optional path to the dataset directory. If not specified,
            a default directory for tabular datasets will be used,
            and datasets will be downloaded if missing.
        exclude_datasets: Optional list of dataset filenames to exclude from the
            benchmark. Each filename should match a file in the dataset directory.
        exclude_image_datasets: Boolean flag that indicates whether to exclude
            image datasets from the benchmark. Defaults to False.
        upper_bound_num_samples: Integer specifying the maximum size of rows
            (in number of samples) to include in the benchmark. Datasets exceeding
            this size will be skipped. Defaults to 10000.
        upper_bound_num_features: Integer specifying the maximum number of features
            to include in the benchmark. Datasets with more features than this
            value will be skipped. Defaults to 500.
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

        result_dir.mkdir(parents=True, exist_ok=True)

    result_df = pl.DataFrame()

    for dataset_file in dataset_paths.glob("*.npz"):
        if dataset_file.name not in exclude_datasets:
            logger.info(f"Running benchmark for {dataset_file.name}...")

            with np.load(dataset_file) as dataset:
                num_samples = dataset["X"].shape[0]
                dataset_name = dataset_file.stem
                num_features = dataset["X"].shape[1]

                if num_samples > upper_bound_num_samples:
                    logger.warning(
                        f"Skipping {dataset_name} "
                        f"- dataset size {num_samples} "
                        f"exceeds limit {upper_bound_num_samples}"
                    )
                    continue
                if num_features > upper_bound_num_features:
                    logger.warning(
                        f"Skipping {dataset_name} "
                        f"- number of features size {num_features} "
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

            outlier_ratio = y.sum() / y.shape[0]

            for embedding_model in embedding_models:
                logger.debug(f"Starting experiment for "
                             f"{embedding_model.name}..."
                             f"Compute Embeddings.")
                if embedding_model.task_only:
                    continue
                try:
                    embeddings, compute_embeddings_time = (
                        embedding_model.generate_embeddings(X, outlier=True)
                    )
                    embed_dim = embeddings.shape[-1]
                except Exception as e:
                    logger.exception(
                        f"By computing embeddings, the following Exception "
                        f"occured: {e}. Skipping"
                    )
                    continue

                logger.debug(
                    f"Start Outlier Detection for {embedding_model.name} "
                )
                for evaluator in evaluators:
                    if evaluator.task_type == "Outlier Detection":
                        prediction, _ = evaluator.get_prediction(
                            embeddings
                        )

                        score_auc = roc_auc_score(y, prediction)

                        evaluator_parameters = evaluator.get_parameters()
                        logger.debug(
                            f"Finished experiment for {evaluator._name} with parameters: "
                            f"{evaluator_parameters}"
                        )
                        new_row_dict = {
                            "dataset_name": [dataset_name],
                            "dataset_size": [num_samples],
                            "embedding_model": [embedding_model.name],
                            "embed_dim": [embed_dim],
                            "algorithm": [evaluator._name],
                            "auc_score": [score_auc],
                            "time_to_compute_train_embedding": [
                                compute_embeddings_time
                            ],
                            "outlier_ratio": [outlier_ratio],
                            "task": ["Outlier Detection"]
                        }

                        for key, value in evaluator_parameters.items():
                            new_row_dict[f"algorithm_{key}"] = [value]

                        new_row = pl.DataFrame(
                            new_row_dict
                        )

                        result_df = pl.concat(
                            [result_df, new_row],
                            how="diagonal"
                        )
                        evaluator.reset_evaluator()
                    else:
                        continue

                logger.debug(
                        f"Finished experiment for {embedding_model.name} and "
                        f"resetting the model."
                    )
                embedding_model.reset_embedding_model()

                if save_result_dataframe:
                    save_result_df(result_df=result_df,
                                   output_path=result_dir,
                                   benchmark_name="ADBench_Tabular",
                                   timestamp=timestamp)

                if get_device() in ["cuda", "mps"]:
                    empty_gpu_cache()

    return result_df
