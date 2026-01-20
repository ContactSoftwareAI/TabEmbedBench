"""Simplified main benchmark orchestration module for TabEmbedBench.

This module provides the main entry point for running comprehensive benchmarks
on embedding models, coordinating outlier detection and supervised evaluations.
"""

import gc
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import polars as pl
import torch

from tabembedbench.benchmark.dataset_separation_benchmark import (
    run_dataseparation_benchmark,
)
from tabembedbench.benchmark.outlier_benchmark import run_outlier_benchmark
from tabembedbench.benchmark.tabarena_benchmark import run_tabarena_benchmark
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.utils.logging_utils import setup_unified_logging, upload_logs_to_gcs


@dataclass
class DatasetConfig:
    """Configuration for specifying dataset settings.

    This class defines the configuration and constraints for datasets used in the application.
    It allows specifying the dataset path, exclusion of specific datasets, and constraints on the
    dataset size and number of features. It also supports selecting specific versions and modes
    (e.g., 'lite') of the dataset.

    Attributes:
        adbench_dataset_path (str | Path | None): Path to the ADBench dataset. If None,
            the dataset path is not specified.
        exclude_adbench_datasets (list[str] | None): List of dataset names to exclude.
            If None, no datasets are excluded.
        tabarena_version (str): Version of the TabArena dataset to use.
        tabarena_lite (bool): Indicates if the 'lite' version of TabArena should be used.
        exclude_tabarena_datasets (list[str] | None): List of dataset names to exclude.
        upper_bound_dataset_size (int): Maximum number of samples allowed in a dataset.
        upper_bound_num_features (int): Maximum number of features allowed in a dataset.
    """

    adbench_dataset_path: str | Path | None = None
    exclude_adbench_datasets: list[str] | None = None
    tabarena_version: str = "tabarena-v0.1"
    tabarena_lite: bool = (True,)
    exclude_tabarena_datasets: list[str] | None = (None,)
    upper_bound_dataset_size: int = 10000
    upper_bound_num_features: int = 500


@dataclass
class BenchmarkConfig:
    """Configuration object for benchmarking tasks.

    This class is used to define the configuration settings for running benchmark
    tests, including flags for specific tasks, directory paths for data storage,
    logging levels, and whether to save logs. It centralizes benchmark-related
    configurations to ensure consistent and customizable behavior across different
    benchmarking runs.

    Attributes:
        run_outlier (bool): Whether to include outlier detection benchmarks in the
            run.
        run_tabarena (bool): Whether to run supervised benchmarks.
        run_tabpfn_subset (bool): Whether to include TabPFN subset benchmarks in
            the run.
        data_dir (str | Path): Path to the directory where necessary data is
            stored or will be stored.
        save_logs (bool): Whether to save logs generated during the benchmarking
            process.
        logging_level (int): The logging level for the benchmark runs, aligning
            with Python's logging module level constants (e.g., logging.INFO).
    """

    run_outlier: bool = True
    run_tabarena: bool = True
    run_dataset_tabpfn_separation: bool = False
    run_dataset_separation: bool = False
    run_tabpfn_subset: bool = False
    data_dir: str | Path = "data"
    save_logs: bool = True
    logging_level: int = logging.INFO
    dataset_separation_configurations_json_path: str | Path | None = None
    dataset_separation_configurations_tabpfn_subset_json_path: str | Path | None = None
    gcs_bucket: str | None = None
    gcs_filepath: str | None = None


def run_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluator_algorithms: list[AbstractEvaluator],
    dataset_config: DatasetConfig | None = None,
    benchmark_config: BenchmarkConfig | None = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, Path]:
    """Run comprehensive benchmark evaluation for embedding models.

    This function orchestrates the complete benchmarking process, running both
    outlier detection and supervised (classification/regression) benchmarks
    on the provided embedding models.

    Args:
        embedding_models: List of embedding model instances to evaluate.
        evaluator_algorithms: List of evaluator instances to use for assessment.
        dataset_config: Configuration for dataset parameters. If None, uses defaults.
        benchmark_config: Configuration for benchmark execution. If None, uses defaults.

    Returns:
        Tuple of (outlier_results_df, tabarena_results_df, result_directory).
    """
    dataset_config = dataset_config or DatasetConfig()
    benchmark_config = benchmark_config or BenchmarkConfig()

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = Path(benchmark_config.data_dir)
    result_dir = data_dir / f"tabembedbench_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file_path = None
    if benchmark_config.save_logs:
        log_dir = result_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = setup_unified_logging(
            log_dir=log_dir,
            timestamp=timestamp,
            logging_level=benchmark_config.logging_level,
            save_logs=benchmark_config.save_logs,
        )

    logger = logging.getLogger("TabEmbedBench_Main")
    logger.info(f"Benchmark started at {datetime.now()}")

    # Validate inputs
    models = _validate_models(embedding_models)
    evaluators = _validate_evaluators(evaluator_algorithms)

    logger.info(f"Using {len(models)} embedding model(s)")
    logger.info(f"Using {len(evaluators)} evaluator(s)")

    if (
        benchmark_config.gcs_bucket is not None
        and benchmark_config.gcs_filepath is not None
    ):
        google_bucket_path = (
            f"{benchmark_config.gcs_bucket}/{benchmark_config.gcs_filepath}"
        )
    else:
        google_bucket_path = None

    # Run outlier detection benchmark
    if benchmark_config.run_outlier:
        logger.info("Running outlier detection benchmark...")
        try:
            result_outlier_df = run_outlier_benchmark(
                embedding_models=models,
                evaluators=evaluators,
                dataset_paths=dataset_config.adbench_dataset_path,
                exclude_datasets=dataset_config.exclude_adbench_datasets,
                upper_bound_num_samples=dataset_config.upper_bound_dataset_size,
                upper_bound_num_features=dataset_config.upper_bound_num_features,
                result_dir=result_dir,
                timestamp=timestamp,
                google_bucket=google_bucket_path,
            )
            _cleanup_models(models, logger)
        except Exception as e:
            logger.exception(f"Error during outlier detection benchmark: {e}")
            result_outlier_df = pl.DataFrame()
    else:
        result_outlier_df = pl.DataFrame()

    # Run TabArena benchmark
    if benchmark_config.run_tabarena:
        logger.info("Running supervised benchmark (TabArena)...")
        try:
            result_tabarena_df = run_tabarena_benchmark(
                embedding_models=models,
                evaluators=evaluators,
                tabarena_version=dataset_config.tabarena_version,
                tabarena_lite=dataset_config.tabarena_lite,
                exclude_datasets=dataset_config.exclude_tabarena_datasets,
                upper_bound_num_samples=dataset_config.upper_bound_dataset_size,
                upper_bound_num_features=dataset_config.upper_bound_num_features,
                timestamp=timestamp,
                result_dir=result_dir,
                run_tabpfn_subset=benchmark_config.run_tabpfn_subset,
                google_bucket=google_bucket_path,
            )
            _cleanup_models(models, logger)
        except Exception as e:
            logger.exception(f"Error during supervised benchmark: {e}")
            result_tabarena_df = pl.DataFrame()
    else:
        logger.info("Skipping supervised benchmark.")
        result_tabarena_df = pl.DataFrame()

    if (
        benchmark_config.run_dataset_tabpfn_separation
        and benchmark_config.dataset_separation_configurations_tabpfn_subset_json_path
        is not None
    ):
        logger.info(
            "Running dataset separation on TabPFN constrainted dataset benchmark..."
        )
        try:
            result_dataset_tabpfn_separation_df = run_dataseparation_benchmark(
                embedding_models=models,
                evaluators=evaluators,
                timestamp=timestamp,
                result_dir=result_dir,
                use_tabpfn_subset=benchmark_config.run_tabpfn_subset,
                dataset_configurations_json_path=benchmark_config.dataset_separation_configurations_tabpfn_subset_json_path,
                google_bucket=google_bucket_path,
            )
            _cleanup_models(models, logger)
        except Exception as e:
            logger.exception(f"Error during supervised benchmark: {e}")
            result_dataset_tabpfn_separation_df = pl.DataFrame()
    else:
        result_dataset_tabpfn_separation_df = pl.DataFrame()

    if (
        benchmark_config.run_dataset_separation
        and benchmark_config.dataset_separation_configurations_json_path is not None
    ):
        logger.info("Running dataset separation benchmark...")
        try:
            result_dataset_separation_df = run_dataseparation_benchmark(
                embedding_models=models,
                evaluators=evaluators,
                timestamp=timestamp,
                result_dir=result_dir,
                use_tabpfn_subset=benchmark_config.run_tabpfn_subset,
                dataset_configurations_json_path=benchmark_config.dataset_separation_configurations_json_path,
                google_bucket=google_bucket_path,
            )
            _cleanup_models(models, logger)
        except Exception as e:
            logger.exception(f"Error during supervised benchmark: {e}")
            result_dataset_separation_df = pl.DataFrame()
    else:
        result_dataset_separation_df = pl.DataFrame()

    if benchmark_config.gcs_bucket and log_file_path:
        logger.info("Uploading logs to Google Cloud Storage...")
        upload_logs_to_gcs(
            local_log_file=log_file_path,
            bucket_name=benchmark_config.gcs_bucket,
            gcs_path=f"{benchmark_config.gcs_filepath}/{str(result_dir)}" or "",
        )

    logger.info(f"Benchmark completed at {datetime.now()}")
    return (
        result_outlier_df,
        result_tabarena_df,
        result_dataset_tabpfn_separation_df,
        result_dataset_separation_df,
        result_dir,
    )


def _validate_models(
    models: list[AbstractEmbeddingGenerator] | AbstractEmbeddingGenerator,
) -> list[AbstractEmbeddingGenerator]:
    """Validate and normalize embedding model inputs.

    Args:
        models: Single model or list of models.

    Returns:
        List of validated AbstractEmbeddingGenerator instances.
    """
    if not isinstance(models, list):
        models = [models]

    return [m for m in models if isinstance(m, AbstractEmbeddingGenerator)]


def _validate_evaluators(
    evaluators: list[AbstractEvaluator] | AbstractEvaluator,
) -> list[AbstractEvaluator]:
    """Validate and normalize evaluator inputs.

    Args:
        evaluators: Single evaluator or list of evaluators.

    Returns:
        List of validated AbstractEvaluator instances.
    """
    if not isinstance(evaluators, list):
        evaluators = [evaluators]

    return [e for e in evaluators if isinstance(e, AbstractEvaluator)]


def _cleanup_models(models: list[AbstractEmbeddingGenerator], logger: logging.Logger):
    """Clean up models and free memory.

    Args:
        models: List of models to clean up.
        logger: Logger for reporting cleanup issues.
    """
    for model in models:
        try:
            model._reset_embedding_model()
        except Exception as e:
            logger.warning(f"Error resetting {model.name}: {e}")

    # Force garbage collection
    gc.collect()

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
