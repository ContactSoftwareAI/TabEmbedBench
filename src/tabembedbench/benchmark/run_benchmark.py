import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import polars as pl

from tabembedbench.benchmark.outlier_benchmark import run_outlier_benchmark
from tabembedbench.benchmark.tabarena_benchmark import run_tabarena_benchmark
from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.logging_utils import setup_unified_logging


def run_benchmark(
    embedding_model: BaseEmbeddingGenerator | None = None,
    embedding_models: list[BaseEmbeddingGenerator] | None = None,
    adbench_dataset_path: str | Path | None = None,
    exclude_adbench_datasets: list[str] | None = None,
    exclude_adbench_image_datasets: bool = True,
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 10000,
    save_embeddings: bool = False,
    run_outlier: bool = True,
    run_task_specific: bool = True,
    data_dir: str | Path = "data",
    save_logs: bool = True,
    logging_level: int = logging.INFO,
    save_result_dataframe: bool = True,
) -> pl.DataFrame:
    """
    Runs benchmarks for outlier detection and task-specific embedding tasks using the
    provided embedding models or model.

    This function orchestrates the benchmarking pipeline for evaluating the performance
    of embedding models on specific tasks or datasets. It includes options to save logs,
    filter datasets, and customize specific functionality such as whether to run outlier
    detection or task-specific benchmarks.

    Args:
        embedding_model (BaseEmbeddingGenerator | None): A single embedding model to be
            evaluated. If provided, only this model will be used for benchmarking. Should
            not be specified together with `embedding_models`.
        embedding_models (list[BaseEmbeddingGenerator] | None): A list of embedding
            models to be evaluated. Should not be specified together with `embedding_model`.
        adbench_dataset_path (str | Path | None): Path to the datasets used for outlier
            detection benchmarks. If `None`, default datasets will be used.
        exclude_adbench_datasets (list[str] | None): List of dataset names to exclude
            from the outlier detection benchmarks. If `None`, no datasets will be excluded.
        exclude_adbench_image_datasets (bool): Whether image-based datasets should be
            excluded from outlier detection benchmarks. Defaults to True.
        tabarena_version (str): TabArena version string, specifying the dataset and
            task versions to be used for task-specific benchmarks. Defaults to
            `"tabarena-v0.1"`.
        tabarena_lite (bool): Whether to perform benchmarks on TabArena-Lite, a smaller
            subset of TabArena. Defaults to True.
        upper_bound_dataset_size (int): Maximum number of data samples to process in any
            single dataset. Defaults to 10000.
        save_embeddings (bool): Whether to save intermediate and final embeddings to disk.
            Defaults to False.
        run_outlier (bool): Whether to perform the outlier detection benchmarks.
            Defaults to True.
        run_task_specific (bool): Whether to perform task-specific benchmarks.
            Defaults to True.
        data_dir (str | Path): Directory where results and logs should be saved.
            Defaults to `"data"`.
        save_logs (bool): Whether to save logs for the benchmarking process. Defaults
            to True.
        logging_level (int): Logging verbosity level. Higher values indicate less logging.
            Defaults to `logging.INFO`.
        save_result_dataframe (bool): Whether to save the benchmarking results as a
            Polars DataFrame as a parquet file on drive. Defaults to True.

    Returns:
        pl.DataFrame: Combined benchmarking results in a Polars DataFrame, containing the
            results for all evaluated models across outlier detection and task-specific
            benchmarks.

    Raises:
        ValueError: If both `embedding_model` and `embedding_models` are specified,
            or if neither is specified.
        ValueError: If any of the models in `embedding_models` lacks the required attributes
            or methods needed for benchmarking.
        ValueError: If `run_outlier` is True but no valid outlier models are provided.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if data_dir is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True)
        log_dir = data_dir / Path("logs")
    else:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        log_dir = data_dir / Path("logs")

    if save_logs:
        setup_unified_logging(
            log_dir=log_dir,
            timestamp=timestamp,
            logging_level=logging_level,
            save_logs=save_logs,
        )

    main_logger = logging.getLogger("TabEmbedBench_Main")
    main_logger.info(f"Benchmark started at {datetime.now()}")

    if embedding_model is None and embedding_models is None:
        raise ValueError("Either model or models must be provided.")
    if embedding_model is not None and embedding_models is not None:
        raise ValueError(
            "Only one of the parameters 'model' or "
            "'models' should be provided, not both."
        )

    if embedding_model is not None:
        models_to_process = [embedding_model]
    else:
        models_to_process = embedding_models

    for model in models_to_process:
        if not hasattr(model, "_compute_embeddings"):
            raise ValueError(
                "There is an element within the list of models "
                "that does not have a function '_compute_embeddings'."
            )
        if not hasattr(model, "name") or not hasattr(model, "_preprocess_data"):
            raise ValueError(
                "There is an element within the list of models "
                "that does not have a property 'name'."
            )

    if run_outlier:
        main_logger.info("Running outlier detection benchmark...")
        outlier_embedding_models = []
        for model in models_to_process:
            if not model.task_only:
                outlier_embedding_models.append(model)
        main_logger.debug(f"Outlier embedding models:"
                         f" {[outlier_embedding_model.name for outlier_embedding_model in outlier_embedding_models]}")
        if len(outlier_embedding_models) > 0:
            main_logger.debug("Start outlier detection function")
            result_outlier_df = run_outlier_benchmark(
                embedding_models=outlier_embedding_models,
                dataset_paths=adbench_dataset_path,
                save_embeddings=save_embeddings,
                exclude_datasets=exclude_adbench_datasets,
                exclude_image_datasets=exclude_adbench_image_datasets,
                upper_bound_num_samples=upper_bound_dataset_size,
                data_dir=data_dir,
                save_result_dataframe=save_result_dataframe,
                timestamp=timestamp,
            )
        else:
            raise ValueError("No outlier models provided.")
    else:
        result_outlier_df = pl.DataFrame()
    if run_task_specific:
        main_logger.info("Running task-specific benchmark (TabArena Lite)...")
        result_tabarena_df = run_tabarena_benchmark(
            embedding_models=models_to_process,
            tabarena_version=tabarena_version,
            tabarena_lite=tabarena_lite,
            upper_bound_dataset_size=upper_bound_dataset_size,
            save_embeddings=save_embeddings,
            timestamp=timestamp,
            data_dir=data_dir,
            save_result_dataframe=save_result_dataframe
        )
    else:
        result_tabarena_df = pl.DataFrame()
        main_logger.info("Skipping task-specific benchmark.")

    result_df = pl.concat(
        [result_outlier_df, result_tabarena_df], how="diagonal"
    )

    if save_result_dataframe:
        main_logger.info("Saving results...")
        result_path = data_dir / "results"
        result_path.mkdir(exist_ok=True)

        result_file = result_path / f"results_{timestamp}.parquet"

        result_df.write_parquet(
            result_file
        )

    return result_df