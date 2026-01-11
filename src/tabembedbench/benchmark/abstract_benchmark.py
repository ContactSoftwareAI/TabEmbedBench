import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
from numpy import ndarray
from tqdm import tqdm

from tabembedbench.constants import TIMESTAMP
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator
from tabembedbench.utils.exception_utils import (
    NotEndToEndCompatibleError,
    NotTaskCompatibleError,
)
from tabembedbench.utils.logging_utils import get_benchmark_logger
from tabembedbench.utils.torch_utils import empty_gpu_cache, get_device, log_gpu_memory
from tabembedbench.utils.tracking_utils import MemoryTracker, save_dataframe
from tabembedbench.utils.unsupervised_metrics import calculate_rank_me


class AbstractBenchmark(ABC):
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
        benchmark_metrics: (
            Dict[str, Dict[str, Callable[[np.ndarray, np.ndarray], float]]] | None
        ) = None,
        gcs_bucket_name: str | None = None,
    ):
        """Initializes an instance of the benchmarking class with specified configurations, logging,
        and optional parameters for task execution.

        Args:
            name (str): Name of the benchmark or task identifier.
            task_type (str | list[str]): Type or types of tasks to execute, e.g., "classification".
            result_dir (str | Path, optional): Directory path where results will be stored. Defaults
                to "results".
            timestamp (str | None, optional): A string timestamp for benchmarking identification.
                If None, the current timestamp is used. Defaults to None.
            logging_level (int, optional): Logging verbosity level. Defaults to logging.INFO.
            save_result_dataframe (bool, optional): Whether to save the results in a DataFrame
                format. Defaults to True.
            upper_bound_num_samples (int, optional): Maximum number of samples allowed during
                task execution. Defaults to 10000.
            upper_bound_num_features (int, optional): Maximum number of features allowed during
                task execution. Defaults to 500.
            benchmark_metrics (Dict[str, Dict[str, Callable[[np.ndarray, np.ndarray], float]]] | None, optional): Dictionary of metrics to evaluate task
                performance. If None, default metrics are used. Defaults to None.
        """
        self.logger = get_benchmark_logger(name)
        self.logger.setLevel(logging_level)
        self._name = name
        self.task_type = task_type if isinstance(task_type, list) else [task_type]
        self.benchmark_metrics = benchmark_metrics or self._get_default_metrics()
        self._results_buffer: list[dict] = []
        self._embedding_metadata_buffer: list[dict] = []
        result_dir = Path(result_dir) if isinstance(result_dir, str) else result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir = result_dir
        self.embeddings_metadata_dir = result_dir / "embeddings"
        self.embeddings_metadata_dir.mkdir(parents=True, exist_ok=True)

        self.bucket_name = gcs_bucket_name
        self.save_to_gcs = True if gcs_bucket_name else False

        self.timestamp = timestamp or TIMESTAMP
        self.save_result_dataframe = save_result_dataframe
        self.upper_bound_num_samples = upper_bound_num_samples
        self.upper_bound_num_features = upper_bound_num_features

    @property
    def name(self) -> str:
        """Gets the name of the benchmark.

        This property retrieves the value of the `_name` attribute,
        which represents the name associated with the benchmark.

        Returns:
            str: The name associated with the instance.
        """
        return self._name

    @property
    def result_df(self) -> pl.DataFrame:
        """Gets the processed result DataFrame.

        This property retrieves a DataFrame representation of data stored in the
        current internal results buffer. If the results buffer is empty, it returns
        an empty DataFrame. The method provides a convenient way to access results
        as a Polars DataFrame.

        Returns:
            pl.DataFrame: A Polars DataFrame instance representing the contents
            of the results buffer. If the results buffer is empty, this returns
            an empty DataFrame.
        """
        if not self._results_buffer:
            return pl.DataFrame()
        return pl.from_dicts(data=self._results_buffer)

    @property
    def embedding_df(self) -> pl.DataFrame:
        """
        Returns a Polars DataFrame representation of the embedding metadata.

        This property retrieves and converts the `_embedding_metadata_buffer` attribute into
        a Polars DataFrame. If the buffer is empty, an empty DataFrame is returned.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the embedding metadata information.
        """
        if not self._embedding_metadata_buffer:
            return pl.DataFrame()
        return pl.from_dicts(data=self._embedding_metadata_buffer)

    # ========== Abstract Methods (Subclasses must implement) ==========
    @abstractmethod
    def _load_datasets(self, **kwargs) -> list[dict[str, Any]]:
        """Load datasets for the benchmark.

        Returns:
            List of datasets to process. Format is benchmark-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def _should_skip_dataset(self, dataset, **kwargs) -> tuple[bool, str]:
        """Determine if a dataset should be skipped.

        Args:
            dataset: The dataset to check.
            **kwargs: Additional parameters for dataset validation.

        Returns:
            bool: True if the dataset should be skipped, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_dataset(self, dataset) -> Iterator[dict]:
        """Prepare data from a dataset for embedding generation.

        This method should yield one or more data splits in a standardized format.
        For outlier detection: yields a single dict with the full dataset.
        For classification/regression: yields multiple dicts for each CV fold.

        Each yielded dict must contain:
            - 'X_train': Training data
            - 'X_test': Test data (can be None)
            - 'y_train': Training labels or None
            - 'y_true': Labels for evaluation (for outlier detection) or test labels for supervised learning
            - 'dataset_metadata': Dictionary containing additional dataset metadata, the following keys should be present:
                - 'dataset_name': Name of the dataset
                - 'num_samples': Number of samples
                - 'num_features': Number of features
            - 'feature_metadata': Dict with any additional info (categorical_indices, etc.)

        Args:
            dataset: The dataset to prepare.

        Yields:
            Dictionary containing prepared data and metadata in standardized format.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_default_metrics(
        self,
    ) -> Dict[str, Dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        """Get the default metrics for the benchmark."""
        raise NotImplementedError

    @abstractmethod
    def _get_evaluator_prediction(
        self,
        embeddings: tuple[np.ndarray, np.ndarray, float],
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> np.ndarray:
        """Generates predictions using the provided evaluator and embeddings according to
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
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Generates embeddings for training and testing datasets using a provided embedding
        model. This function ensures that the embedding generation process is handled in a
        model-specific manner while accommodating configurations such as feature metadata.

        Args:
            embedding_model: An instance of AbstractEmbeddingGenerator that performs the
                embedding generation process for the datasets.
            dataset_configurations: A dictionary containing configuration details for the
                datasets. Must include the key "X_train". Optionally, it may also include
                the keys "X_test" and "feature_metadata".

        Returns:
            Tuple[np.ndarray, np.ndarray | None, float]: A tuple where:
                - The first element is the embeddings for the training dataset (X_train).
                - The second element is the embeddings for the testing dataset (X_test) if
                  provided; otherwise, it is None.
                - The third element is a float that indicates some additional contextual metric
                  or information calculated during the embedding generation process.
        """
        X_train = dataset_configurations.get("X_train")
        X_test = dataset_configurations.get("X_test")

        # Pass feature_metadata for model-specific handling
        feature_metadata = dataset_configurations.get("feature_metadata", {})

        return embedding_model.generate_embeddings(
            X_train=X_train,
            X_test=X_test,
            outlier=("Outlier Detection" in self.task_type),
            **feature_metadata,
        )

    def _embedding_utils(
        self,
        embeddings: tuple[np.ndarray, np.ndarray, dict],
        model_memory: dict,
        dataset_configurations: dict,
    ) -> None:
        train_embeddings, test_embeddings, embedding_metadata = embeddings
        dataset_metadata = dataset_configurations["dataset_metadata"]
        embedding_metadata.update(model_memory)
        embedding_metadata["train_rank_me_score"] = calculate_rank_me(
            embeddings=train_embeddings
        )
        embedding_metadata["test_rank_me_score"] = (
            calculate_rank_me(embeddings=test_embeddings)
            if test_embeddings is not None
            else None
        )
        embedding_metadata.update(dataset_metadata)
        self._embedding_metadata_buffer.append(embedding_metadata)
        self._save_embedding_metadata(dataframe_name="embedding_metadata")
        return None

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str
    ) -> dict[str, float]:
        """Computes and returns a dictionary of evaluation metrics for the given task type by applying
        the specified metric functions on the provided true and predicted values.

        Args:
            y_true: Ground truth labels. Structure and format depend on the specific task.
            y_pred: Predicted labels or scores produced by the model.
                Structure and format depend on the specific task.
            task_type: Type of the task to be evaluated (e.g., "classification", "regression").
                Determines which metrics and functions to apply.

        Returns:
            Dict[str, float]: A dictionary containing the task type and the computed metrics.
                The dictionary keys are metric names, and the values are their corresponding
                computed values.
        """
        metric_scores = {"task": task_type}

        for metric in self.benchmark_metrics[task_type]:
            metric_func: Callable[[ndarray, ndarray], float] = self.benchmark_metrics[
                task_type
            ][metric]
            metric_scores[metric] = metric_func(y_true, y_pred)

        return metric_scores

    def _process_embedding_model_pipeline(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        evaluators: list[AbstractEvaluator],
        dataset_configurations: dict,
    ) -> None:
        """Processes an embedding model over a given dataset with a set of evaluators. Generates
        embeddings for the provided dataset using the embedding model, evaluates the generated
        embeddings with compatible evaluators, and saves the results.

        Args:
            embedding_model: Embedding model to be processed.
            evaluators: List of evaluators to assess the generated embeddings.
            dataset_configurations: A subset of the dataset to be processed with the embedding model.

        """
        dataset_metadata = dataset_configurations["dataset_metadata"]
        logger_prefix = f"Dataset: {dataset_metadata['dataset_name']} - Embedding Model: {embedding_model.name}"

        self.logger.info(f"{logger_prefix} - Start processing...")
        log_gpu_memory(self.logger)

        result_row_dict = {
            "embedding_model": embedding_model.name,
            "dataset_name": dataset_metadata["dataset_name"],
        }

        # Generate embeddings
        try:
            self.logger.info(f"{logger_prefix} - Generating embeddings...")

            model_memory_tracker = MemoryTracker()
            model_memory_tracker.start_tracking()

            embeddings = self._generate_embeddings(
                embedding_model, dataset_configurations
            )
            model_memory = model_memory_tracker.stop_tracking()

            self._embedding_utils(embeddings, model_memory, dataset_configurations)
        except Exception as e:
            self.logger.exception(f"{logger_prefix} - Error generating embeddings: {e}")
            raise

        for evaluator in evaluators:
            current_result = result_row_dict.copy()
            if not self._is_compatible(evaluator, dataset_metadata.get("task_type")):
                self.logger.debug(
                    f"{logger_prefix} - Skipping evaluator {evaluator.name} is not compatible with {self.task_type}. Skipping..."
                )
                continue
            self.logger.info(
                f"{logger_prefix} - Evaluating embeddings with {evaluator.name}..."
            )
            prediction = self._get_evaluator_prediction(
                embeddings,
                evaluator,
                dataset_configurations,
            )
            metric_scores = self._compute_metrics(
                y_true=dataset_configurations["y_true"],
                y_pred=prediction,
                task_type=dataset_metadata.get("task_type"),
            )
            current_result["algorithm"] = evaluator.name
            current_result.update(metric_scores)
            current_result.update(evaluator.get_parameters())

            evaluator.reset_evaluator()
            self._cleanup_gpu_cache()
            self._results_buffer.append(current_result)

        # Save intermediate results after each model
        self._save_results()
        self._cleanup_gpu_cache()

    def _process_end_to_end_model_pipeline(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        dataset_configurations: dict,
    ) -> None:
        """Processes the pipeline for an end-to-end model.

        This function is intended to handle the processing related to end-to-end
        models but is not implemented. It raises a NotImplementedError when called
        to indicate that the functionality is not supported by the class.

        Args:
            embedding_model: Instance of AbstractEmbeddingGenerator responsible for
                generating embeddings for the data.
            dataset_configurations: Dictionary containing data split information, typically with
                keys that specify different parts of the dataset (e.g., 'train',                'validation', 'test') and their associated data.

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
        """Processes the dataset configuration by applying embedding models and evaluators.
        Determines if an embedding model is an end-to-end model and processes it accordingly.

        Args:
            dataset_configurations (dict): Dictionary containing dataset configuration details.
            embedding_models (list[AbstractEmbeddingGenerator]): List of embedding model instances
                to be applied on the dataset.
            evaluators (list[AbstractEvaluator]): List of evaluator instances to validate the
                embedding models on the dataset.

        Raises:
            NotEndToEndCompatibleError: Raised when an end-to-end incompatible embedding model is
                encountered. This exception is logged and processing continues with the next model.
            Exception: Handles all other unforeseen exceptions that occur during embedding model
                processing. These exceptions are logged along with the relevant dataset and
                embedding model details.
        """
        for embedding_model in embedding_models:
            logger_prefix = f"Dataset: {dataset_configurations['dataset_metadata']['dataset_name']} - Embedding Model: {embedding_model.name}"
            try:
                if not embedding_model.check_dataset_constraints(
                    num_samples=dataset_configurations["dataset_metadata"][
                        "num_samples"
                    ],
                    num_features=dataset_configurations["dataset_metadata"][
                        "num_features"
                    ],
                ):
                    self.logger.info(
                        f"{logger_prefix} - Skipping embedding model due to size constraints."
                    )
                    continue
                if embedding_model.is_end_to_end_model:
                    self._process_end_to_end_model_pipeline(
                        embedding_model, dataset_configurations
                    )
                else:
                    self._process_embedding_model_pipeline(
                        embedding_model, evaluators, dataset_configurations
                    )
            except NotEndToEndCompatibleError as e:
                self.logger.info(str(e))
                continue
            except NotTaskCompatibleError as e:
                self.logger.info(str(e))
                continue
            except Exception as e:
                self.logger.exception(
                    f"Error processing embedding model {embedding_model.name} on dataset {dataset_configurations['dataset_metadata'].get('dataset_name', 'Unknown')}: {e}"
                )
                continue

    def _process_dataset(
        self,
        dataset,
        embedding_models: list[AbstractEmbeddingGenerator],
        evaluators: list[AbstractEvaluator],
    ) -> None:
        """Processes a given dataset through multiple embedding models and evaluators.

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
        """
        should_skip, msg = self._should_skip_dataset(dataset)
        if should_skip:
            self.logger.info(msg)
            return

        # Prepare dataset (may yield multiple splits for cross-validation)
        try:
            self.logger.info(msg)
            dataset_configurations = self._prepare_dataset(dataset)
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

        datasets_to_process = (
            datasets.values() if isinstance(datasets, dict) else datasets
        )

        for dataset in tqdm(
            datasets_to_process,
            desc=f"Running {self.name} benchmark. Processing datasets...",
        ):
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

            save_dataframe(
                dataframe=sorted_df,
                output_path=self.result_dir,
                dataframe_name=f"result_{self.name}",
                timestamp=self.timestamp,
                save_to_gcs=self.save_to_gcs,  # ✓ Present
                bucket_name=self.bucket_name,
            )

    def _save_embedding_metadata(self, dataframe_name: str) -> None:
        """Save the current results to disk if enabled."""
        if self.save_result_dataframe and not self.embedding_df.is_empty():
            save_dataframe(
                dataframe=self.embedding_df,
                output_path=self.embeddings_metadata_dir,
                dataframe_name=f"{self.name}_embedding_{dataframe_name}",
                timestamp=self.timestamp,
                save_to_gcs=self.save_to_gcs,
                bucket_name=self.bucket_name,
            )

    def _cleanup_gpu_cache(self):
        """Clear GPU cache if available."""
        if get_device() in ["cuda", "mps"]:
            empty_gpu_cache()
