import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import polars as pl

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device, log_gpu_memory
from tabembedbench.utils.tracking_utils import save_result_df


class NotEndToEndCompatibleError(Exception):
    """Raised when the embedding model is not an end-to-end model."""

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.message = (
            f"The benchmark {self.benchmark} is not compatible for end to end models."
        )
        super().__init__(self.message)


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
        name: str,
        task_type: str | list[str],
        result_dir: str | Path = "results",
        timestamp: str | None = None,
        logging_level: int = logging.INFO,
        save_result_dataframe: bool = True,
        upper_bound_num_samples: int = 10000,
        upper_bound_num_features: int = 500,
        benchmark_metrics: Dict | None = None,
    ):
        """
        Initializes the benchmark object with the provided parameters and sets up
        necessary configurations such as logging, metrics, and result directories.

        Args:
            name (str): The name of the benchmark.
            task_type (str | list[str]): The type of task, either a string or a list
                of task types.
            result_dir (str | Path, optional): The directory where results will be saved.
                Defaults to "results".
            timestamp (str | None, optional): A specific timestamp for the benchmark
                execution. If not provided, the current timestamp is used. Defaults
                to None.
            save_result_dataframe (bool, optional): Whether to save the result
                dataframe. Defaults to True.
            upper_bound_num_samples (int, optional): The upper limit on the number
                of samples for benchmarking. Defaults to 10000.
            upper_bound_num_features (int, optional): The upper limit on the number
                of features for benchmarking. Defaults to 500.
            benchmark_metrics (Dict | None, optional): A dictionary of benchmark
                metrics to be used. If not provided, default metrics will be used.
                Defaults to None.
        """
        self.logger = get_benchmark_logger(name)
        self.logger.setLevel(logging_level)
        self._name = name
        self.task_type = task_type if isinstance(task_type, list) else [task_type]
        self.benchmark_metrics = benchmark_metrics or self._get_default_metrics()
        self._results_buffer: List[dict] = []
        result_dir = Path(result_dir) if isinstance(result_dir, str) else result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir = result_dir

        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_result_dataframe = save_result_dataframe
        self.upper_bound_num_samples = upper_bound_num_samples
        self.upper_bound_num_features = upper_bound_num_features

    @property
    def name(self) -> str:
        """Gets the name attribute value.

        This property retrieves the value of the `_name` attribute,
        which represents the name associated with the instance.

        Returns:
            str: The name associated with the instance.
        """
        return self._name

    @property
    def result_df(self):
        if not self._results_buffer:
            return pl.DataFrame()
        return pl.from_dicts(data=self._results_buffer)

    # ========== Abstract Methods (Subclasses must implement) ==========
    @abstractmethod
    def _load_datasets(self, **kwargs) -> List[Dict[str, Any]]:
        """Load datasets for the benchmark.

        Returns:
            List of datasets to process. Format is benchmark-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def _should_skip_dataset(self, dataset, **kwargs) -> Tuple[bool, str]:
        """Determine if a dataset should be skipped.

        Args:
            dataset: The dataset to check.
            **kwargs: Additional parameters for dataset validation.

        Returns:
            bool: True if the dataset should be skipped, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_dataset(self, dataset, **kwargs) -> Iterator[dict]:
        """Prepare data from a dataset for embedding generation.

        This method should yield one or more data splits in a standardized format.
        For outlier detection: yields a single dict with the full dataset.
        For classification/regression: yields multiple dicts for each CV fold.

        Each yielded dict must contain:
            - 'X_train': Training data or None
            - 'X_test': Test data or None
            - 'y_train': Training labels or None
            - 'y_eval': Labels for evaluation (for outlier detection) or test labels for supervised learning
            - 'dataset_metadata': Dictionary containing additional dataset metadata, such as:
                - 'dataset_name': Name of the dataset
                - 'dataset_size': Number of samples
                - 'num_features': Number of features
            - 'feature_metadata': Dict with any additional info (categorical_indices, etc.)

        Args:
            dataset: The dataset to prepare.
            **kwargs: Additional parameters for data preparation.

        Yields:
            Dictionary containing prepared data and metadata in standardized format.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_default_metrics(
        self,
    ) -> dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        """Get the default metrics for the benchmark."""
        raise NotImplementedError

    @abstractmethod
    def _get_evaluator_prediction(
        self,
        embeddings: Tuple[np.ndarray, np.ndarray, float],
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> np.ndarray:
        """
        Generates predictions using the provided evaluator and embeddings according to
        the specified dataset configurations.

        Args:
            embeddings: Tuple containing the embeddings to be evaluated.
            evaluator: An instance of AbstractEvaluator that computes predictions.
            dataset_configurations: Dictionary specifying dataset-related configurations.

        Returns:
            np.ndarray: An array of predictions generated by the evaluator.

        Raises:
            NotImplementedError: Always raised, as this method is abstract and must
            be implemented by a subclass.
        """
        raise NotImplementedError

    # ========== Concrete Methods (Implemented in base class) ==========
    def _generate_embeddings(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        dataset_configurations: dict,
    ) -> tuple:
        """Generate embeddings using the provided model.

        Args:
            embedding_model: The embedding model to use.
            dataset_configurations: Dictionary containing data and feature_metadata.

        Returns:
            Tuple of (train_embeddings, test_embeddings, compute_time).
        """
        X_train = dataset_configurations.get("X_train")
        X_test = dataset_configurations.get("X_test", None)

        # Pass feature_metadata for model-specific handling
        feature_metadata = dataset_configurations.get("feature_metadata", {})

        return embedding_model.generate_embeddings(
            X_train=X_train,
            X_test=X_test,
            outlier=(self.task_type == "Outlier Detection"),
            **feature_metadata,
        )

    def _compute_metrics(self, y_true, y_pred, task_type: str) -> dict:
        """
        Computes evaluation metrics for the provided predictions and ground truth
        values based on the specified task type.

        The function calculates metrics using pre-defined metric functions for
        the specified task type. Results are compiled into a dictionary
        containing the task type and computed metric values.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            task_type: A string representing the type of task for which metrics
                are being computed, such as "classification" or "regression".

        Returns:
            A dictionary containing the task type and the computed metric values
            for the given task.
        """
        result_dict = {"task_type": task_type}

        for metric in self.benchmark_metrics[task_type]:
            metric_func = self.benchmark_metrics[task_type][metric]
            result_dict[metric] = metric_func(y_true, y_pred)

        return result_dict

    def _process_embedding_model(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        evaluators: List[AbstractEvaluator],
        dataset_configurations: dict,
    ) -> None:
        """
        Processes an embedding model over a given dataset with a set of evaluators. Generates
        embeddings for the provided dataset using the embedding model, evaluates the generated
        embeddings with compatible evaluators, and saves the results.

        Args:
            embedding_model: Embedding model to be processed.
            evaluators: List of evaluators to assess the generated embeddings.
            dataset_configurations: A subset of the dataset to be processed with the embedding model.

        """
        dataset_metadata = dataset_configurations["dataset_metadata"]
        self.logger.info(
            f"Start processing {embedding_model.name} on {dataset_metadata['dataset_name']}..."
        )
        log_gpu_memory(self.logger)
        self.logger.info(
            f"Processing {embedding_model.name} on {dataset_metadata['dataset_name']}..."
        )

        result_row_dict = {"embedding_model": embedding_model.name}
        result_row_dict.update(dataset_metadata)

        # Generate embeddings
        try:
            self.logger.info(f"Generating embeddings with {embedding_model.name}...")
            embeddings = self._generate_embeddings(
                embedding_model, dataset_configurations
            )
        except Exception as e:
            self.logger.exception(
                f"Error generating embeddings with {embedding_model.name}: {e}"
            )
            raise

        for evaluator in evaluators:
            if not self._is_compatible(evaluator, dataset_metadata.get("task_type")):
                self.logger.info(
                    f"Skipping evaluator {evaluator.name} is not compatible with {self.task_type}. Skipping..."
                )
                continue
            else:
                self.logger.info(f"Evaluating embeddings with {evaluator.name}...")
                prediction = self._get_evaluator_prediction(
                    embeddings,
                    dataset_configurations["y_train"],
                    evaluator,
                    dataset_configurations,
                )
                metric_scores = self._compute_metrics(
                    y_true=dataset_configurations["y_true"],
                    y_pred=prediction,
                    task_type=dataset_metadata.get("task_type"),
                )
                result_row_dict.update(metric_scores)
                result_row_dict.update(evaluator.get_parameters())

            evaluator.reset_evaluator()
            self._cleanup_gpu_cache()
            self._results_buffer.append(result_row_dict)

        # Save intermediate results after each model
        self._save_results()
        self._cleanup_gpu_cache()

    def _process_end_to_end_model_pipeline(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        dataset_configurations: dict,
    ) -> None:
        """
        Processes the pipeline for an end-to-end model.

        This function is intended to handle the processing related to end-to-end
        models but is not implemented. It raises a NotImplementedError when called
        to indicate that the functionality is not supported by the class.

        Args:
            embedding_model: Instance of AbstractEmbeddingGenerator responsible for
                generating embeddings for the data.
            dataset_configurations: Dictionary containing data split information, typically with
                keys that specify different parts of the dataset (e.g., 'train',
                'validation', 'test') and their associated data.

        Raises:
            NotImplementedError: Indicates that this functionality is not supported
                by the class.
        """
        raise NotEndToEndCompatibleError(benchmark=self.name)

    def _process_dataset_configuration(
        self,
        dataset_configurations: dict,
        embedding_models: list[AbstractEmbeddingGenerator],
        evaluators: list[AbstractEvaluator],
    ) -> None:
        """Process all embedding models on a single data split."""
        for embedding_model in embedding_models:
            try:
                if embedding_model.is_end_to_end_model:
                    self._process_end_to_end_model_pipeline(
                        embedding_model, dataset_configurations
                    )
                else:
                    self._process_embedding_model(
                        embedding_model, evaluators, dataset_configurations
                    )
            except NotEndToEndCompatibleError as e:
                self.logger.info(str(e))
                continue
            except Exception as e:
                self.logger.exception(
                    f"Error processing embedding model {embedding_model.name} on dataset {dataset_configurations['dataset_metadata'].get('dataset_name', 'Unknown')}: {e}"
                )
                continue

    def _validate_dataset_configurations(
        self, dataset_configurations: Iterator[dict]
    ) -> None:
        pass

    def _process_dataset(
        self,
        dataset,
        embedding_models: list[AbstractEmbeddingGenerator],
        evaluators: list[AbstractEvaluator],
        **kwargs,
    ) -> None:
        """
        Processes a given dataset through multiple embedding models and evaluators.

        This function determines if the dataset should be skipped based on specific
        conditions. If it is not skipped, the dataset is prepared (potentially
        yielding multiple splits for tasks like cross-validation) and then processed
        using the provided embedding models and evaluators.

        Args:
            dataset: The dataset to be processed. Format and structure are expected to
                match the requirements of the embedding models and evaluators.
            embedding_models (list[AbstractEmbeddingGenerator]): A list of embedding
                model instances responsible for generating embeddings for the dataset.
            evaluators (list[AbstractEvaluator]): A list of evaluator instances used
                to evaluate the dataset's embeddings or other features.
            **kwargs: Additional configuration options for dataset preparation and
                processing, dependent on the specific implementation details of the
                embedding models or evaluators.
        """
        should_skip, msg = self._should_skip_dataset(dataset, **kwargs)
        if should_skip:
            self.logger.info(msg)
            return

        # Prepare dataset (may yield multiple splits for cross-validation)
        try:
            self.logger.info(msg)
            dataset_configurations = self._prepare_dataset(dataset, **kwargs)
        except Exception as e:
            self.logger.exception(f"Error preparing dataset: {e}")
            return

        # Process each data split
        for dataset_configuration in dataset_configurations:
            self._process_dataset_configuration(
                dataset_configuration, embedding_models, evaluators
            )

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
        self.logger.info(f"Starting {self.name} benchmark...")

        datasets = self._load_datasets(**kwargs)
        # TODO: Counter machen.
        for dataset in datasets:
            self._process_dataset(dataset, embedding_models, evaluators, **kwargs)

        self.logger.info(f"{self.name} benchmark completed.")
        return self.result_df

    def _is_compatible(
        self, evaluator: AbstractEvaluator, dataset_tasktype: str
    ) -> bool:
        """Check if evaluator is compatible with the current task.

        Args:
            evaluator: The evaluator to check.
            data_split: Dictionary containing data split information.

        Returns:
            True if evaluator is compatible, False otherwise.
        """
        # Check task type compatibility
        return dataset_tasktype in evaluator.task_type

    def _check_dataset_size_constraints(
        self, num_samples: int, num_features: int, dataset_name: str
    ) -> list[str] | list[None]:
        """Check if dataset exceeds size constraints.

        Args:
            num_samples: Number of samples in the dataset.
            num_features: Number of features in the dataset.
            dataset_name: Name of the dataset for logging.

        Returns:
            Tuple of (should_skip: bool, reason: str | None).
        """
        skip_reasons = []

        if num_samples > self.upper_bound_num_samples:
            skip_reasons.append(
                f"Too many samples ({num_samples} > {self.upper_bound_num_samples})"
            )

        if num_features > self.upper_bound_num_features:
            skip_reasons.append(
                f"Too many features ({num_features} > {self.upper_bound_num_features})"
            )

        return skip_reasons

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
                benchmark_name=self.name,
                timestamp=self.timestamp,
            )

    def _cleanup_gpu_cache(self):
        """Clear GPU cache if available."""
        if get_device() in ["cuda", "mps"]:
            empty_gpu_cache()
