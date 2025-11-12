from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Iterator

import polars as pl

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device, log_gpu_memory
from tabembedbench.utils.tracking_utils import save_result_df


class AbstractBenchmark(ABC):
    """Abstract base class for benchmark implementations.

    This class provides a simplified structure for different benchmark types
    (e.g., outlier detection, classification, regression). It handles common
    functionality such as result management, logging, and GPU cache management.

    The workflow is clear and linear:
    1. Load datasets
    2. For each dataset:
       a. Check if should skip
       b. Prepare data (yields one or more data splits)
       c. For each data split:
          - Generate embeddings with each model
          - Evaluate embeddings with each evaluator
          - Save results

    Attributes:
        logger: Logger instance for the benchmark.
        result_df: Polars DataFrame to store benchmark results.
        result_dir: Directory path for saving results.
        timestamp: Timestamp string for result file naming.
        save_result_dataframe: Flag to determine if results should be saved.
        upper_bound_num_samples: Maximum number of samples to process.
        upper_bound_num_features: Maximum number of features to process.
        task_type: Type of task (e.g., 'Outlier Detection', 'Supervised Classification').
    """

    def __init__(
        self,
        logger_name: str,
        task_type: str,
        result_dir: str | Path = "results",
        timestamp: str | None = None,
        save_result_dataframe: bool = True,
        upper_bound_num_samples: int = 10000,
        upper_bound_num_features: int = 500,
    ):
        """Initialize the abstract benchmark.

        Args:
            logger_name: Name for the logger instance.
            task_type: Type of task for this benchmark.
            result_dir: Directory path for saving results.
            timestamp: Optional timestamp string. If None, current time is used.
            save_result_dataframe: Whether to save results to disk.
            upper_bound_num_samples: Maximum dataset size to process.
            upper_bound_num_features: Maximum number of features to process.
        """
        self.logger = get_benchmark_logger(logger_name)
        self.task_type = task_type
        self.result_df = pl.DataFrame()

        if isinstance(result_dir, str):
            result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir = result_dir

        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_result_dataframe = save_result_dataframe
        self.upper_bound_num_samples = upper_bound_num_samples
        self.upper_bound_num_features = upper_bound_num_features

    # ========== Abstract Methods (Subclasses must implement) ==========

    @abstractmethod
    def _load_datasets(self, **kwargs) -> list:
        """Load datasets for the benchmark.

        Returns:
            List of datasets to process. Format is implementation-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def _should_skip_dataset(self, dataset, **kwargs) -> tuple[bool, str | None]:
        """Determine if a dataset should be skipped.

        Args:
            dataset: The dataset to check.
            **kwargs: Additional parameters for dataset validation.

        Returns:
            Tuple of (should_skip: bool, reason: str | None).
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_dataset(self, dataset, **kwargs) -> Iterator[dict]:
        """Prepare data from a dataset for embedding generation.

        This method should yield one or more data splits in a standardized format.
        For outlier detection: yields a single dict with the full dataset.
        For classification/regression: yields multiple dicts for each CV fold.

        Each yielded dict must contain:
            - 'X': Full data (for outlier detection) or None
            - 'X_train': Training data or None
            - 'X_test': Test data or None
            - 'y': Full labels (for outlier detection) or None
            - 'y_train': Training labels or None
            - 'y_test': Test labels or None
            - 'dataset_name': Name of the dataset
            - 'dataset_size': Number of samples
            - 'num_features': Number of features
            - 'metadata': Dict with any additional info (categorical_indices, etc.)

        Args:
            dataset: The dataset to prepare.
            **kwargs: Additional parameters for data preparation.

        Yields:
            Dictionary containing prepared data and metadata in standardized format.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate(
        self,
        embeddings: tuple,
        evaluator: AbstractEvaluator,
        data_split: dict,
    ) -> dict:
        """Evaluate embeddings using the provided evaluator.

        Args:
            embeddings: Tuple from embedding generation (train_emb, test_emb, time).
            evaluator: The evaluator instance to use.
            data_split: Dictionary containing the data split and metadata.

        Returns:
            Dictionary containing evaluation results (must include all metadata).
        """
        raise NotImplementedError

    @abstractmethod
    def _get_benchmark_name(self) -> str:
        """Get the name of the benchmark for result saving.

        Returns:
            String identifier for the benchmark type.
        """
        raise NotImplementedError

    # ========== Concrete Methods (Implemented in base class) ==========

    def run_benchmark(
        self,
        embedding_models: list[AbstractEmbeddingGenerator],
        evaluators: list[AbstractEvaluator],
        **kwargs,
    ) -> pl.DataFrame:
        """Run the benchmark with the provided models and evaluators.

        This is the main entry point with a clear, linear workflow.

        Args:
            embedding_models: List of embedding models to evaluate.
            evaluators: List of evaluators to use.
            **kwargs: Additional benchmark-specific parameters.

        Returns:
            Polars DataFrame containing all benchmark results.
        """
        self.logger.info(f"Starting {self._get_benchmark_name()} benchmark...")

        datasets = self._load_datasets(**kwargs)

        for dataset in datasets:
            # Check if dataset should be skipped
            should_skip, skip_reason = self._should_skip_dataset(dataset, **kwargs)
            if should_skip:
                self.logger.warning(skip_reason)
                continue

            # Prepare dataset (may yield multiple splits for cross-validation)
            try:
                data_splits = self._prepare_dataset(dataset, **kwargs)
            except Exception as e:
                self.logger.exception(f"Error preparing dataset: {e}")
                continue

            # Process each data split
            for data_split in data_splits:
                # Process each embedding model
                for embedding_model in embedding_models:
                    log_gpu_memory(self.logger)
                    self.logger.info(
                        f"Processing {embedding_model.name} on {data_split['dataset_name']}..."
                    )

                    # Generate embeddings
                    try:
                        embeddings = self._generate_embeddings(
                            embedding_model, data_split
                        )
                    except Exception as e:
                        self.logger.exception(
                            f"Error generating embeddings with {embedding_model.name}: {e}"
                        )
                        continue

                    # Evaluate with each compatible evaluator
                    for evaluator in evaluators:
                        if not self._is_compatible(evaluator, data_split):
                            continue

                        try:
                            results = self._evaluate(embeddings, evaluator, data_split)
                            # Add embedding model name to results
                            results["embedding_model"] = [embedding_model.name]
                            self._add_result(results)
                        except Exception as e:
                            self.logger.exception(
                                f"Error evaluating with {evaluator._name}: {e}"
                            )

                        # Reset evaluator for next use
                        evaluator.reset_evaluator()
                        self._cleanup_gpu_cache()

                    # Save intermediate results after each model
                    self._save_results()
                    self._cleanup_gpu_cache()

        self.logger.info(f"{self._get_benchmark_name()} benchmark completed.")
        return self.result_df

    def _generate_embeddings(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        data_split: dict,
    ) -> tuple:
        """Generate embeddings using the provided model.

        Args:
            embedding_model: The embedding model to use.
            data_split: Dictionary containing data and metadata.

        Returns:
            Tuple of (train_embeddings, test_embeddings, compute_time).
        """
        # Extract data based on task type
        if data_split.get("X") is not None:
            # Outlier detection: single dataset
            X_train = data_split["X"]
            X_test = None
        else:
            # Supervised learning: train/test split
            X_train = data_split["X_train"]
            X_test = data_split["X_test"]

        # Pass metadata for model-specific handling
        metadata = data_split.get("metadata", {})

        return embedding_model.generate_embeddings(
            X_train=X_train,
            X_test=X_test,
            outlier=(self.task_type == "Outlier Detection"),
            **metadata,
        )

    def _is_compatible(self, evaluator: AbstractEvaluator, data_split: dict) -> bool:
        """Check if evaluator is compatible with the current task.

        Args:
            evaluator: The evaluator to check.
            data_split: Dictionary containing data split information.

        Returns:
            True if evaluator is compatible, False otherwise.
        """
        # Check task type compatibility
        return evaluator.task_type == self.task_type

    def _check_dataset_size_constraints(
        self, num_samples: int, num_features: int, dataset_name: str
    ) -> tuple[bool, str | None]:
        """Check if dataset exceeds size constraints.

        Args:
            num_samples: Number of samples in the dataset.
            num_features: Number of features in the dataset.
            dataset_name: Name of the dataset for logging.

        Returns:
            Tuple of (should_skip: bool, reason: str | None).
        """
        if num_samples > self.upper_bound_num_samples:
            reason = (
                f"Skipping {dataset_name} - dataset size {num_samples} "
                f"exceeds limit {self.upper_bound_num_samples}"
            )
            return True, reason

        if num_features > self.upper_bound_num_features:
            reason = (
                f"Skipping {dataset_name} - number of features {num_features} "
                f"exceeds limit {self.upper_bound_num_features}"
            )
            return True, reason

        return False, None

    def _add_result(self, result_dict: dict):
        """Add a result row to the result DataFrame.

        Args:
            result_dict: Dictionary containing result data.
        """
        new_row = pl.DataFrame(result_dict)
        self.result_df = pl.concat([self.result_df, new_row], how="diagonal")

    def _save_results(self):
        """Save the current results to disk if enabled."""
        if self.save_result_dataframe and not self.result_df.is_empty():
            # Sort columns: non-algorithm columns first, then algorithm columns
            non_algo_cols = [
                col for col in self.result_df.columns if not col.startswith("algorithm")
            ]
            algo_cols = [
                col for col in self.result_df.columns if col.startswith("algorithm")
            ]

            sorted_df = self.result_df.select(non_algo_cols + algo_cols)

            save_result_df(
                result_df=sorted_df,
                output_path=self.result_dir,
                benchmark_name=self._get_benchmark_name(),
                timestamp=self.timestamp,
            )

    def _cleanup_gpu_cache(self):
        """Clear GPU cache if available."""
        if get_device() in ["cuda", "mps"]:
            empty_gpu_cache()
