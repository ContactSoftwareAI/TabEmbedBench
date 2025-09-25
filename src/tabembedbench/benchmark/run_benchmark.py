from contextlib import contextmanager
from datetime import datetime
import gc
import logging
from pathlib import Path

import polars as pl
import torch

from tabembedbench.benchmark.outlier_benchmark import run_outlier_benchmark
from tabembedbench.benchmark.tabarena_benchmark import run_tabarena_benchmark
from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator
from tabembedbench.utils.logging_utils import setup_unified_logging


@contextmanager
def benchmark_context(models_to_process, main_logger, context_name="benchmark"):
    """
    Provides a context manager to benchmark and manage the lifecycle of a collection
    of models. The context logs the start and end of the process, ensures embedded
    models are reset, clears memory, and optionally flushes GPU memory. Useful for
    efficient and clean benchmarking of models.

    Args:
        models_to_process (List[Any]): Collection of models to process within
            the context.
        main_logger (logging.Logger): Logger instance for logging context lifecycle
            events and warnings.
        context_name (str, optional): Name for the context to appear in the logs.
            Defaults to "benchmak".

    Yields:
        List[Any]: The same list of models provided in 'models_to_process'.
    """
    try:
        main_logger.info(f"Starting {context_name} at {datetime.now()}")
        yield models_to_process
    finally:
        main_logger.info(f"Cleaning up {context_name} at {datetime.now()}")
        for model in models_to_process:
            try:
                model.reset_embedding_model()
            except Exception as e:
                main_logger.warning(f"Error occurred during resetting "
                                    f"{model.name}: {e}")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        main_logger.info(f"Finished {context_name} at {datetime.now()}")


def run_benchmark(
    embedding_model: AbstractEmbeddingGenerator | None = None,
    embedding_models: list[AbstractEmbeddingGenerator] | None = None,
    adbench_dataset_path: str | Path | None = None,
    exclude_adbench_datasets: list[str] | None = None,
    exclude_adbench_image_datasets: bool = True,
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 10000,
    upper_bound_num_feautres: int = 500,
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
        embedding_model (AbstractEmbeddingGenerator | None): A single embedding model to be
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
        upper_bound_num_feautres (int): Maximum number of features to process in any
            single dataset. Defaults to 500.
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
        result_dir = Path(data_dir / f"tabembedbench_{timestamp}")
        result_dir.mkdir(exist_ok=True)
        log_dir = Path(result_dir / "logs")
        log_dir.mkdir(exist_ok=True)
    else:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        result_dir = Path(data_dir / f"tabembedbench_{timestamp}")
        result_dir.mkdir(exist_ok=True)
        log_dir = Path(result_dir / "logs")
        log_dir.mkdir(exist_ok=True)

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
        main_logger.debug(
            f"Outlier embedding models:"
            f" {[model.name for model in outlier_embedding_models]}"
        )
        if len(outlier_embedding_models) > 0:
            main_logger.debug("Start outlier detection benchmark")
            try:
                with benchmark_context(outlier_embedding_models, main_logger,
                                       "ADBench Outlier Detection"):
                    result_outlier_df = run_outlier_benchmark(
                        embedding_models=outlier_embedding_models,
                        dataset_paths=adbench_dataset_path,
                        save_embeddings=save_embeddings,
                        exclude_datasets=exclude_adbench_datasets,
                        exclude_image_datasets=exclude_adbench_image_datasets,
                        upper_bound_num_samples=upper_bound_dataset_size,
                        upper_bound_num_features=upper_bound_num_feautres,
                        result_dir=result_dir,
                        save_result_dataframe=save_result_dataframe,
                        timestamp=timestamp,
                    )
            except Exception as e:
                main_logger.error(f"Error occurred during "
                                  f"outlier detection benchmark: {e}")
                result_outlier_df = pl.DataFrame()
        else:
            raise ValueError("No outlier models provided.")
    else:
        result_outlier_df = pl.DataFrame()
    if run_task_specific:
        main_logger.info("Running task-specific benchmark (TabArena Lite)...")
        try:
            with benchmark_context(models_to_process, main_logger, "TabArena Lite"):
                result_tabarena_df = run_tabarena_benchmark(
                    embedding_models=models_to_process,
                    tabarena_version=tabarena_version,
                    tabarena_lite=tabarena_lite,
                    upper_bound_num_samples=upper_bound_dataset_size,
                    upper_bound_num_features=upper_bound_num_feautres,
                    save_embeddings=save_embeddings,
                    timestamp=timestamp,
                    result_dir=result_dir,
                    save_result_dataframe=save_result_dataframe
                )
        except Exception as e:
            main_logger.error(f"Error occurred during task-specific benchmark: {e}")
            result_tabarena_df = pl.DataFrame()

    else:
        result_tabarena_df = pl.DataFrame()
        main_logger.info("Skipping task-specific benchmark.")

    result_df = pl.concat(
        [result_outlier_df, result_tabarena_df], how="diagonal"
    )

    return result_df
