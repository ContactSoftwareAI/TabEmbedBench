"""Main benchmark orchestration module for TabEmbedBench.

This module provides the main entry point for running comprehensive benchmarks
on embedding models, coordinating outlier detection and task-specific evaluations.
"""

import gc
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Tuple

import polars as pl
import torch

from tabembedbench.benchmark.outlier_benchmark import run_outlier_benchmark
from tabembedbench.benchmark.tabarena_benchmark import run_tabarena_benchmark
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.utils.logging_utils import setup_unified_logging


@contextmanager
def benchmark_context(models_to_process, main_logger, context_name="benchmark"):
    """Provides a context manager to benchmark and manage the lifecycle of a collection
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
                model._reset_embedding_model()
            except Exception as e:
                main_logger.warning(
                    f"Error occurred during resetting {model.name}: {e}"
                )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        main_logger.info(f"Finished {context_name} at {datetime.now()}")


def run_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluator_algorithms: list[AbstractEvaluator],
    tabarena_specific_embedding_models: list[AbstractEmbeddingGenerator] | None = None,
    adbench_dataset_path: str | Path | None = None,
    exclude_adbench_datasets: list[str] | None = None,
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_dataset_size: int = 10000,
    upper_bound_num_features: int = 500,
    run_outlier: bool = True,
    run_task_specific: bool = True,
    data_dir: str | Path = "data",
    save_logs: bool = True,
    run_tabpfn_subset: bool = False,
    logging_level=logging.INFO,
) -> Tuple[pl.DataFrame, pl.DataFrame, Path]:
    """Run comprehensive benchmark evaluation for embedding models.

    This function orchestrates the complete benchmarking process, running both
    outlier detection and task-specific (classification/regression) benchmarks
    on the provided embedding models. It handles result collection, logging,
    and resource management.

    Args:
        embedding_models (list[AbstractEmbeddingGenerator]): List of embedding model
            instances to evaluate.
        evaluator_algorithms (list[AbstractEvaluator]): List of evaluator instances
            to use for assessment.
        adbench_dataset_path (str | Path | None, optional): Path to ADBench datasets.
            If None, uses default path. Defaults to None.
        exclude_adbench_datasets (list[str] | None, optional): List of ADBench dataset
            filenames to exclude. Defaults to None.
        tabarena_version (str, optional): OpenML TabArena suite version identifier.
            Defaults to "tabarena-v0.1".
        tabarena_lite (bool, optional): Whether to use lite mode for faster execution
            with fewer cross-validation folds. Defaults to True.
        upper_bound_dataset_size (int, optional): Maximum number of samples to process.
            Datasets exceeding this will be skipped. Defaults to 10000.
        upper_bound_num_features (int, optional): Maximum number of features to process.
            Datasets exceeding this will be skipped. Defaults to 500.
        run_outlier (bool, optional): Whether to run outlier detection benchmark.
            Defaults to True.
        run_task_specific (bool, optional): Whether to run TabArena task-specific
            benchmark. Defaults to True.
        data_dir (str | Path, optional): Directory for saving results and logs.
            Defaults to "data".
        save_logs (bool, optional): Whether to save logs to file. Defaults to True.
        logging_level (int, optional): Logging verbosity level. Defaults to logging.INFO.

    Returns:
        pl.DataFrame: Polars DataFrame containing combined results from all benchmarks,
            including performance metrics, timing information, and model parameters.
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

    models_to_process = validate_embedding_models(embedding_models)

    if tabarena_specific_embedding_models is not None:
        tabarena_models_to_process = validate_embedding_models(
            tabarena_specific_embedding_models
        )
    else:
        tabarena_models_to_process = []

    evaluators_to_use = validate_evaluator_models(evaluator_algorithms)

    main_logger.info(f"Using {len(models_to_process)} embedding model(s)")
    main_logger.info(f"Using {len(evaluators_to_use)} evaluator(s)")

    if run_outlier:
        main_logger.info("Running outlier detection benchmark...")
        try:
            with benchmark_context(
                models_to_process, main_logger, "ADBench Outlier Detection"
            ):
                result_outlier_df = run_outlier_benchmark(
                    embedding_models=models_to_process,
                    evaluators=evaluators_to_use,
                    dataset_paths=adbench_dataset_path,
                    exclude_datasets=exclude_adbench_datasets,
                    upper_bound_num_samples=upper_bound_dataset_size,
                    upper_bound_num_features=upper_bound_num_features,
                    result_dir=result_dir,
                    timestamp=timestamp,
                )
        except Exception as e:
            main_logger.exception(
                f"Error occurred during outlier detection benchmark: {e}"
            )
            result_outlier_df = pl.DataFrame()
    else:
        result_outlier_df = pl.DataFrame()
    if run_task_specific:
        main_logger.info("Running task-specific benchmark (TabArena Lite)...")
        embedding_models_to_process = tabarena_models_to_process + models_to_process
        try:
            with benchmark_context(models_to_process, main_logger, "TabArena Lite"):
                result_tabarena_df = run_tabarena_benchmark(
                    embedding_models=embedding_models_to_process,
                    evaluators=evaluators_to_use,
                    tabarena_version=tabarena_version,
                    tabarena_lite=tabarena_lite,
                    upper_bound_num_samples=upper_bound_dataset_size,
                    upper_bound_num_features=upper_bound_num_features,
                    timestamp=timestamp,
                    result_dir=result_dir,
                    run_tabpfn_subset=run_tabpfn_subset,
                )
        except Exception as e:
            main_logger.exception(f"Error occurred during task-specific benchmark: {e}")
            result_tabarena_df = pl.DataFrame()

    else:
        result_tabarena_df = pl.DataFrame()
        main_logger.info("Skipping task-specific benchmark.")

    return result_outlier_df, result_tabarena_df, result_dir


def validate_embedding_models(embedding_models):
    """Validate and filter embedding model instances.

    Args:
        embedding_models: Single model instance or list of model instances.

    Returns:
        list: List of validated AbstractEmbeddingGenerator instances.
    """
    if not isinstance(embedding_models, list):
        models_to_check = [embedding_models]
    else:
        models_to_check = embedding_models

    return [
        model
        for model in models_to_check
        if isinstance(model, AbstractEmbeddingGenerator)
    ]


def validate_evaluator_models(evaluator_algorithms):
    """Validate and filter evaluator algorithm instances.

    Args:
        evaluator_algorithms: Single evaluator instance or list of evaluator instances.

    Returns:
        list: List of validated AbstractEvaluator instances.
    """
    if not isinstance(evaluator_algorithms, list):
        algorithms_to_check = [evaluator_algorithms]
    else:
        algorithms_to_check = evaluator_algorithms

    return [
        algorithm
        for algorithm in algorithms_to_check
        if isinstance(algorithm, AbstractEvaluator)
    ]
