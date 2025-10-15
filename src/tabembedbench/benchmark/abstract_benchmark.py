from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import polars as pl

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device
from tabembedbench.utils.tracking_utils import save_result_df


class AbstractBenchmark(ABC):
    """Abstract base class for benchmark implementations.

    This class provides a common structure for different benchmark types
    (e.g., outlier detection, classification, regression). It handles common
    functionality such as result management, logging, GPU cache management,
    and the overall benchmark execution flow.

    Attributes:
        logger: Logger instance for the benchmark.
        result_df: Polars DataFrame to store benchmark results.
        result_dir: Directory path for saving results.
        timestamp: Timestamp string for result file naming.
        save_result_dataframe: Flag to determine if results should be saved.
        upper_bound_num_samples: Maximum number of samples to process.
        upper_bound_num_features: Maximum number of features to process.
    """

    def __init__(
        self,
        logger_name: str,
        result_dir: str | Path = "results",
        timestamp: str | None = None,
        save_result_dataframe: bool = True,
        upper_bound_num_samples: int = 10000,
        upper_bound_num_features: int = 500,
    ):
        """Initialize the abstract benchmark.

        Args:
            logger_name: Name for the logger instance.
            result_dir: Directory path for saving results.
            timestamp: Optional timestamp string. If None, current time is used.
            save_result_dataframe: Whether to save results to disk.
            upper_bound_num_samples: Maximum dataset size to process.
            upper_bound_num_features: Maximum number of features to process.
        """
        self.logger = get_benchmark_logger(logger_name)
        self.result_df = pl.DataFrame()

        if isinstance(result_dir, str):
            result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir = result_dir

        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_result_dataframe = save_result_dataframe
        self.upper_bound_num_samples = upper_bound_num_samples
        self.upper_bound_num_features = upper_bound_num_features

    @abstractmethod
    def _load_datasets(self, **kwargs):
        """Load datasets for the benchmark.

        This method should be implemented by subclasses to handle
        dataset-specific loading logic (e.g., loading from files,
        downloading from OpenML, etc.).

        Returns:
            A list of datasets to process. Each dataset can be any format
            (file path, dict, object) as long as it's compatible with
            _should_skip_dataset() and _prepare_data().
        """
        raise NotImplementedError

    @abstractmethod
    def _should_skip_dataset(self, dataset, **kwargs) -> tuple[bool, str | None]:
        """Determine if a dataset should be skipped.

        Args:
            dataset: The dataset to check.
            **kwargs: Additional parameters for dataset validation.

        Returns:
            A tuple of (should_skip: bool, reason: str | None).
            If should_skip is True, reason contains the skip message.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_data(self, dataset, **kwargs):
        """Prepare data from a dataset for embedding generation.

        This method should extract and preprocess the data from the dataset
        into a format suitable for the embedding models.

        Args:
            dataset: The dataset to prepare.
            **kwargs: Additional parameters for data preparation.

        Returns:
            Prepared data in the format expected by embedding models.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate_embeddings(
        self, embeddings, evaluator: AbstractEvaluator, dataset_info: dict, **kwargs
    ) -> dict:
        """Evaluate embeddings using the provided evaluator.

        Args:
            embeddings: The embeddings to evaluate.
            evaluator: The evaluator instance to use.
            dataset_info: Dictionary containing dataset metadata.
            **kwargs: Additional parameters for evaluation.

        Returns:
            Dictionary containing evaluation results.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_benchmark_name(self) -> str:
        """Get the name of the benchmark for result saving.

        Returns:
            String identifier for the benchmark type.
        """
        raise NotImplementedError

    def _check_dataset_size_constraints(
        self, num_samples: int, num_features: int, dataset_name: str
    ) -> tuple[bool, str | None]:
        """Check if dataset exceeds size constraints.

        Args:
            num_samples: Number of samples in the dataset.
            num_features: Number of features in the dataset.
            dataset_name: Name of the dataset for logging.

        Returns:
            A tuple of (should_skip: bool, reason: str | None).
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

    def _generate_embeddings(
        self, embedding_model: AbstractEmbeddingGenerator, prepared_data,
            **kwargs
    ):
        """Generate embeddings using the provided model.

        Args:
            embedding_model: The embedding model to use.
            data: The data to generate embeddings for.
            **kwargs: Additional parameters for embedding generation.

        Returns:
            Tuple of (embeddings, compute_time, test_embeddings, test_compute_time).

        Raises:
            Exception: If embedding generation fails.
        """
        data = prepared_data["data"]
        embedding_kwargs = prepared_data["embedding_kwargs"]
        try:
            return embedding_model.generate_embeddings(data, **embedding_kwargs)
        except Exception as e:
            self.logger.exception(
                f"Error computing embeddings with {embedding_model.name}: {e}"
            )
            raise

    def _add_result_row(self, result_dict: dict):
        """Add a result row to the result DataFrame.

        Args:
            result_dict: Dictionary containing result data.
        """
        new_row = pl.DataFrame(result_dict)
        self.result_df = pl.concat([self.result_df, new_row], how="diagonal")

    def _save_results(self):
        """Save the current results to disk if enabled."""
        if self.save_result_dataframe and not self.result_df.is_empty():
            save_result_df(
                result_df=self.result_df,
                output_path=self.result_dir,
                benchmark_name=self._get_benchmark_name(),
                timestamp=self.timestamp,
            )

    def _cleanup_gpu_cache(self):
        """Clear GPU cache if available."""
        if get_device() in ["cuda", "mps"]:
            empty_gpu_cache()

    def _normalize_prepared_data(self, prepared_data):
        """Normalize prepared data to an iterable format.

        Args:
            prepared_data: Data from _prepare_data, can be a dict, generator, or list.

        Returns:
            An iterable of prepared data items.
        """
        # Handle both single dict and generator cases
        # If it's a generator, iterate through it; otherwise, wrap in a list
        try:
            # Try to iterate (works for generators and lists)
            if hasattr(prepared_data, "__iter__") and not isinstance(
                prepared_data, dict
            ):
                return iter(prepared_data)
            else:
                return [prepared_data]
        except TypeError:
            # If not iterable, wrap in a list
            return [prepared_data]

    def _process_embedding_generation(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        prepared_data_item: dict
    ) -> tuple | None:
        """Generate embeddings for a prepared data item.

        Args:
            embedding_model: The embedding model to use.
            prepared_data_item: Dictionary containing prepared data and metadata.

        Returns:
            Tuple of (embedding_results, embed_dim) if successful, None otherwise.
        """
        try:
            embedding_results = self._generate_embeddings(
                embedding_model,
                prepared_data_item,
            )

            return embedding_results
        except Exception as e:
            self.logger.exception(f"Error generating embeddings: {e}. Skipping.")
            # Add error row if dataset info is available
            if "dataset_name" in prepared_data_item:
                error_row = {
                    "dataset_name": [prepared_data_item["dataset_name"]],
                    "dataset_size": [prepared_data_item.get("dataset_size", 0)],
                    "embedding_model": [embedding_model.name],
                }
                self._add_result_row(error_row)
            return None

    def _process_single_evaluation(
        self,
        evaluator: AbstractEvaluator,
        embedding_results,
        prepared_data_item: dict,
        embedding_model: AbstractEmbeddingGenerator,
        embed_dim: int,
    ):
        """Process a single evaluation with an evaluator.

        Args:
            evaluator: The evaluator to use.
            embedding_results: Results from embedding generation.
            prepared_data_item: Dictionary containing prepared data and metadata.
            embedding_model: The embedding model used.
            embed_dim: Dimension of the embeddings.
        """
        if not self._is_evaluator_compatible(
            evaluator, **prepared_data_item.get("eval_kwargs", {})
        ):
            return

        self.logger.debug(f"Starting evaluation with {evaluator._name}...")

        try:
            embeddings = embedding_results[0]
            time_to_compute_embedding = embedding_results[1]
            # Prepare dataset info for evaluation
            dataset_info = {
                "dataset_name": [prepared_data_item["dataset_name"]],
                "dataset_size": [prepared_data_item["dataset_size"]],
                "num_features": [prepared_data_item["num_features"]],
                "embedding_model": [embedding_model.name],
                "embed_dim": [embed_dim],
                "time_to_compute_embedding": [time_to_compute_embedding],
                "algorithm": [evaluator._name],
            }

            # Evaluate embeddings
            eval_results = self._evaluate_embeddings(
                embeddings,
                evaluator,
                dataset_info,
                **prepared_data_item.get("eval_kwargs", {}),
            )

            # Add evaluator parameters to results
            evaluator_params = evaluator.get_parameters()
            for key, value in evaluator_params.items():
                eval_results[f"algorithm_{key}"] = [value]

            # Add result row
            self._add_result_row(eval_results)

            # Reset evaluator
            evaluator.reset_evaluator()

            self.logger.debug(f"Finished evaluation with {evaluator._name}")

        except Exception as e:
            self.logger.exception(
                f"Error during evaluation with {evaluator._name}: {e}"
            )

    def _process_embedding_model(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        prepared_data_item: dict,
        evaluators: list[AbstractEvaluator],
    ):
        """Process a single embedding model for a prepared data item.

        Args:
            embedding_model: The embedding model to use.
            prepared_data_item: Dictionary containing prepared data and metadata.
            evaluators: List of evaluators to use.
        """
        self.logger.info(f"Starting experiment for {embedding_model.name}...")

        # Generate embeddings
        result = self._process_embedding_generation(embedding_model, prepared_data_item)
        if result is None:
            return

        embedding_results = result

        embed_dim = embedding_results[0].shape[-1]

        # Evaluate with each evaluator
        for evaluator in evaluators:
            self._process_single_evaluation(
                evaluator,
                embedding_results,
                prepared_data_item,
                embedding_model,
                embed_dim,
            )
            self._cleanup_gpu_cache()

        # Save intermediate results
        self._save_results()

        # Cleanup
        self._cleanup_gpu_cache()

        self.logger.debug(f"Finished experiment for {embedding_model.name}")

    def _process_dataset(
        self,
        dataset,
        embedding_models: list[AbstractEmbeddingGenerator],
        evaluators: list[AbstractEvaluator],
        **kwargs,
    ):
        """Process a single dataset with all embedding models and evaluators.

        Args:
            dataset: The dataset to process.
            embedding_models: List of embedding models to evaluate.
            evaluators: List of evaluators to use.
            **kwargs: Additional benchmark-specific parameters.
        """
        # Check if dataset should be skipped
        should_skip, skip_reason = self._should_skip_dataset(dataset, **kwargs)
        if should_skip:
            self.logger.warning(skip_reason)
            return

        # Prepare data from dataset
        try:
            prepared_data = self._prepare_data(dataset, **kwargs)
        except Exception as e:
            self.logger.exception(f"Error preparing data: {e}")
            return

        # Normalize prepared data to iterable format
        data_items = self._normalize_prepared_data(prepared_data)

        # Process each data item (could be one or many)
        for prepared_data_item in data_items:
            # Process each embedding model
            for embedding_model in embedding_models:
                self._process_embedding_model(
                    embedding_model, prepared_data_item, evaluators
                )

    def run_benchmark(
        self,
        embedding_models: list[AbstractEmbeddingGenerator],
        evaluators: list[AbstractEvaluator],
        **kwargs,
    ) -> pl.DataFrame:
        """Run the benchmark with the provided models and evaluators.

        This is the main entry point for executing the benchmark. It iterates
        through datasets, generates embeddings, evaluates them, and collects
        results.

        Args:
            embedding_models: List of embedding models to evaluate.
            evaluators: List of evaluators to use.
            **kwargs: Additional benchmark-specific parameters.

        Returns:
            Polars DataFrame containing all benchmark results.
        """
        datasets = self._load_datasets(**kwargs)

        for dataset in datasets:
            self._process_dataset(dataset, embedding_models, evaluators, **kwargs)

        self.logger.info(f"{self._get_benchmark_name()} benchmark completed.")
        return self.result_df

    def _is_evaluator_compatible(self, evaluator: AbstractEvaluator, **kwargs) -> bool:
        """Check if evaluator is compatible with the current benchmark.

        This method can be overridden by subclasses to implement
        benchmark-specific compatibility checks.

        Args:
            evaluator: The evaluator to check.
            **kwargs: Additional parameters for compatibility checking.

        Returns:
            True if evaluator is compatible, False otherwise.
        """
        return True
