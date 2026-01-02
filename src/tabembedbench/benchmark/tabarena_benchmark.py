import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import openml
import pandas as pd
import polars as pl
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.benchmark.constants import (
    SUPERVISED_BINARY_CLASSIFICATION,
    SUPERVISED_MULTICLASSIFICATION,
    SUPERVISED_REGRESSION,
    TABARENA_TABPFN_SUBSET,
)
from tabembedbench.embedding_models.abstractembedding import (
    AbstractEmbeddingGenerator,
)
from tabembedbench.evaluators.abstractevaluator import AbstractEvaluator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class TabArenaBenchmark(AbstractBenchmark):
    """Simplified benchmark for TabArena classification and regression tasks.

    This benchmark evaluates embedding models on supervised learning tasks
    from the OpenML TabArena benchmark suite.
    """

    def __init__(
        self,
        tabarena_version: str = "tabarena-v0.1",
        tabarena_lite: bool = True,
        exclude_datasets: list[str] | None = None,
        result_dir: str | Path = "result_tabarena",
        timestamp: str = TIMESTAMP,
        logging_level: int = logging.INFO,
        save_result_dataframe: bool = True,
        upper_bound_num_samples: int = 100000,
        upper_bound_num_features: int = 500,
        run_tabpfn_subset: bool = True,
        skip_missing_values: bool = True,
        benchmark_metrics: dict | None = None,
        openml_cache_dir: str | Path | None = None,
    ):
        """
        Initializes the class with configuration options for running benchmarks using TabArena datasets.
        The class is designed to support various supervised learning tasks with configurable parameters
        such as benchmark versions, dataset inclusion/exclusion, and cache management. Additionally,
        it sets up directories and logging for efficient benchmark execution.

        Args:
            tabarena_version (str): Defines the version of the TabArena benchmark suite used.
            tabarena_lite (bool): Determines whether to use the lightweight configuration of TabArena.
            exclude_datasets (list[str] | None): List of datasets to exclude from the benchmark process.
            result_dir (str | Path): Path to the directory where benchmark results will be stored.
            timestamp (str): Timestamp used for organizing run-specific outputs and logs.
            logging_level (int): Logging level for the benchmark process (e.g., logging.INFO).
            save_result_dataframe (bool): Determines whether to save benchmark results as a DataFrame.
            upper_bound_num_samples (int): Maximum allowed number of samples in the dataset.
            upper_bound_num_features (int): Maximum allowed number of features in the dataset.
            run_tabpfn_subset (bool): Indicates if a specific subset of TabPFN should be included.
            skip_missing_values (bool): Specifies whether to skip datasets with missing values.
            benchmark_metrics (dict | None): Dictionary of benchmark metrics to use. Defaults to a
                predefined set.
            openml_cache_dir (str | Path | None): Path to the directory for caching OpenML datasets.
                Defaults to "data/tabarena_datasets" if not provided.
        """
        super().__init__(
            name="TabEmbedBench_TabArena",
            task_type=[
                SUPERVISED_REGRESSION,
                SUPERVISED_BINARY_CLASSIFICATION,
                SUPERVISED_MULTICLASSIFICATION,
            ],  # Will be overridden per split
            result_dir=result_dir,
            timestamp=timestamp,
            logging_level=logging_level,
            save_result_dataframe=save_result_dataframe,
            upper_bound_num_samples=upper_bound_num_samples,
            upper_bound_num_features=upper_bound_num_features,
        )

        self.tabarena_version = tabarena_version
        self.tabarena_lite = tabarena_lite
        self.run_tabpfn_subset = run_tabpfn_subset
        self.benchmark_suite = None
        self.task_ids = None
        self.len_tabpfn_subset = len(TABARENA_TABPFN_SUBSET)
        self.exclude_datasets = exclude_datasets or []
        self.skip_missing_values = skip_missing_values
        self.benchmark_metrics = benchmark_metrics or self._get_default_metrics()

        if openml_cache_dir is None:
            openml_cache_dir = Path("data/tabarena_datasets")
        else:
            openml_cache_dir = Path(openml_cache_dir)

        openml_cache_dir.mkdir(parents=True, exist_ok=True)
        openml.config.set_root_cache_directory(openml_cache_dir)
        self.openml_cache_dir = openml_cache_dir

        self.logger.info(f"OpenML cache directory: {self.openml_cache_dir}")

    def _load_datasets(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Loads and processes datasets from the OpenML benchmark suite.

        This method retrieves the OpenML benchmark suite based on the specified
        `tabarena_version`, extracts tasks from the suite, and fetches their corresponding
        datasets. Additionally, it determines the folds and repeats configuration for each
        task dataset and compiles the collected information into a list.

        Args:
            **kwargs: Optional keyword arguments that may be used in the dataset loading
                process.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the following
                information for a task:
                - task_id: The identifier of the task.
                - task: The OpenML task object.
                - dataset: The dataset associated with the task.
                - folds: The number of folds for cross-validation associated with the task.
                - repeats: The number of repeats for cross-validation associated with the task.
        """
        self.benchmark_suite = openml.study.get_suite(self.tabarena_version)
        self.task_ids = self.benchmark_suite.tasks

        datasets = []
        for task_id in self.task_ids:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            folds, repeats = self._get_task_configuration(dataset, task)

            datasets.append(
                {
                    "task_id": task_id,
                    "task": task,
                    "dataset": dataset,
                    "folds": folds,
                    "repeats": repeats,
                }
            )

        return datasets

    def _should_skip_dataset(self, dataset_info: dict, **kwargs) -> Tuple[bool, str]:
        """Check if a dataset should be skipped.

        Args:
            dataset_info: Dictionary containing task and dataset information.
            **kwargs: Additional parameters (unused).

        Returns:
            Tuple of (should_skip, reason).
        """
        task_id = dataset_info["task_id"]
        dataset = dataset_info["dataset"]
        num_samples = dataset.qualities["NumberOfInstances"]
        num_features = dataset.qualities["NumberOfFeatures"]
        num_of_missing_values = dataset.qualities["NumberOfMissingValues"]

        skip_reasons: list[str] = []

        skip_reasons.extend(
            self._check_dataset_size_constraints(
                num_samples, num_features, dataset.name
            )
        )

        if num_of_missing_values > 0 and self.skip_missing_values:
            skip_reasons.append("Contains missing values")

        if self.run_tabpfn_subset and task_id not in TABARENA_TABPFN_SUBSET:
            skip_reasons.append("Not in TabPFN subset")

        if dataset.name in self.exclude_datasets:
            skip_reasons.append("Excluded by user")

        if skip_reasons:
            reason = " | ".join(skip_reasons)
            return True, f"Dataset {dataset.name} - Skipping dataset: {reason}"

        self.len_tabpfn_subset -= 1
        task = dataset_info["task"]
        msg = (
            f"Dataset {dataset.name} - Starting experiments for task {task.task_type}. "
            f"{self.len_tabpfn_subset} datasets remaining."
        )

        return False, msg

    def _prepare_dataset(self, dataset_info: dict) -> Iterator[dict]:
        """
        Prepares the dataset for model training and evaluation by performing preprocessing
        and creating train-test splits. Handles categorical and numerical transformations,
        removes irrelevant features, and encodes classification labels when necessary.

        The method processes data fold by fold and repeat by repeat, yielding independent
        data splits for each fold and repeat combination.

        Args:
            dataset_info (dict): A dictionary containing dataset meta-information.
                - "task": The task object containing task specifications such as
                  task type ("Supervised Classification") and target_name.
                - "dataset": The dataset object from which data can be retrieved.
                - "folds": The number of data folds for splitting.
                - "repeats": The number of repeats for each fold.

        Yields:
            Iterator[dict]: An iterator that generates dictionaries, each containing
                processed data, metadata, and train-test splits for a specific fold/repeat.
                - "X_train", "X_test" (DataFrame): Transformed training and test input data.
                - "y_train", "y_test" (Series): Encoded training and test target labels.
                - "dataset_name" (str): The name of the dataset.
                - "dataset_size" (int): Total number of records in the dataset.
                - "num_features" (int): Number of features in the dataset.
                - "metadata" (dict): Additional information about the task and processing,
                  such as task type, categorical columns, fold, and repeat.

        Raises:
            Any exception raised during data retrieval, preprocessing, or train-test splitting
            will propagate to the caller. Ensure proper exception handling is implemented
            where this function is used.
        """
        task = dataset_info["task"]
        dataset = dataset_info["dataset"]
        folds = dataset_info["folds"]
        repeats = dataset_info["repeats"]

        dataset_metadata = {
            "dataset_name": dataset.name,
            "num_samples": dataset.qualities.get("NumberOfInstances"),
            "num_features": dataset.qualities.get("NumberOfFeatures"),
            "num_classes": dataset.qualities.get("NumberOfClasses", None),
            "percentage_of_numeric_features": dataset.qualities.get(
                "PercentageNumericFeatures", None
            ),
            "ratio_features_samples": dataset.qualities["Dimensionality"],
        }

        task_type = task.task_type

        X, y, categorical_indicator, _ = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )

        X, categorical_indicator = self._remove_columns_with_one_unique_value(
            X,
            categorical_indicator,
            dataset.name,
        )

        # Iterate through all folds and repeats
        for repeat in range(repeats):
            for fold in range(folds):
                # Get train/test split
                train_indices, test_indices = task.get_train_test_split_indices(
                    fold=fold,
                    repeat=repeat,
                )

                categorical_column_names = [
                    col
                    for col, is_categorical in zip(X.columns, categorical_indicator)
                    if is_categorical
                ]
                categorical_indices = [
                    i
                    for i, is_categorical in enumerate(categorical_indicator)
                    if is_categorical
                ]

                X_train = X.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_train = y.iloc[train_indices]
                y_test = y.iloc[test_indices]

                # Encode labels for classification
                if task_type == "Supervised Classification":
                    label_encoder = LabelEncoder()

                    y_train = label_encoder.fit_transform(y_train)
                    y_test = label_encoder.transform(y_test)
                    task_type = (
                        "Supervised Multiclass Classification"
                        if (dataset_metadata["num_classes"] > 2)
                        else "Supervised Binary Classification"
                    )

                dataset_metadata["task_type"] = task_type

                yield {
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_true": y_test,
                    "dataset_metadata": dataset_metadata,
                    "feature_metadata": {
                        "categorical_indices": categorical_indices,
                        "categorical_column_names": categorical_column_names,
                        "fold": fold,
                        "repeat": repeat,
                    },
                }

    def _get_default_metrics(
        self,
    ) -> dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        """
        Returns default metrics for different supervised learning tasks.

        This function provides a mapping of supervised learning task types to their default
        evaluation metrics. The supported task types and their associated metrics are as
        follows:
        - SUPERVISED_REGRESSION: Includes "mape_score" for Mean Absolute Percentage Error.
        - SUPERVISED_BINARY_CLASSIFICATION: Includes "auc_score" for Area Under the ROC Curve.
        - SUPERVISED_MULTICLASSIFICATION: Includes "auc_score" for Area Under the ROC Curve
          with one-vs-rest strategy and "log_loss_score" for Log Loss.

        Returns:
            dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]: A nested dictionary
            where keys represent supervised task types, and values are dictionaries that map
            metric names to callable functions implementing those metrics.
        """
        return {
            SUPERVISED_REGRESSION: {
                "mae_score": mean_absolute_error,
                "mape_score": mean_absolute_percentage_error,
                "r2_score": r2_score,
                "rmse_score": root_mean_squared_error,
            },
            SUPERVISED_BINARY_CLASSIFICATION: {
                "auc_score": roc_auc_score,
            },
            SUPERVISED_MULTICLASSIFICATION: {
                "auc_score_ovr": partial(roc_auc_score, multi_class="ovr"),
                "auc_score_ovo": partial(roc_auc_score, multi_class="ovo"),
                "log_loss_score": log_loss,
            },
        }

    def _get_evaluator_prediction(
        self,
        embeddings: Tuple[np.ndarray, np.ndarray, float],
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> np.ndarray:
        """
        Generates evaluator predictions based on provided embeddings and dataset configurations.

        This function is responsible for using the provided embeddings and dataset
        configurations to train the evaluator and generate predictions for the test
        embeddings. It supports different task types, such as supervised binary
        classification, and formats the output predictions accordingly.

        Args:
            embeddings (Tuple[np.ndarray, np.ndarray, float]): A tuple containing the train
                embeddings, test embeddings, and a float value (usually an objective
                metric or loss value).
            evaluator (AbstractEvaluator): An instance of the evaluator that provides
                prediction capabilities. The evaluator should implement a `get_prediction`
                method for training and test inference.
            dataset_configurations (dict): A dictionary containing dataset-configured
                details required for the training and prediction process. It includes
                metadata like `task_type` and data like `y_train`.

        Returns:
            np.ndarray: The predictions generated by the evaluator for the test embeddings.
        """
        task_type = dataset_configurations["dataset_metadata"]["task_type"]
        train_embeddings, test_embeddings, _ = embeddings
        y_train = dataset_configurations["y_train"]

        # Train evaluator
        evaluator.get_prediction(
            train_embeddings,
            y_train,
            train=True,
        )

        # Get test predictions
        test_prediction, _ = evaluator.get_prediction(
            test_embeddings,
            train=False,
        )

        if task_type == SUPERVISED_BINARY_CLASSIFICATION:
            test_prediction = test_prediction[:, 1]

        return test_prediction

    def _process_end_to_end_model_pipeline(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        dataset_configurations: dict,
    ) -> None:
        """Processes an end-to-end model pipeline to generate predictions and compute evaluation metrics.

        This function manages the pipeline for training a given end-to-end model on the specified dataset,
        generating predictions for a test set, computing relevant metrics, and storing the results in a
        buffer for later usage.

        Args:
            embedding_model: An instance of AbstractEmbeddingGenerator that supplies methods for embedding
                generation and end-to-end prediction.
            dataset_configurations: A dictionary containing configurations of the dataset, including
                training and test data, ground truth labels, dataset metadata, and feature metadata. The
                following keys are expected in the dictionary:
                - X_train: Training features.
                - y_train: Training labels.
                - X_test: Testing features.
                - y_true: Ground truth labels for the test data.
                - dataset_metadata: A dictionary describing dataset-specific metadata, such as task type.
                - feature_metadata: A dictionary describing feature-related metadata, such as current fold
                  and repeat.

        Returns:
            None
        """
        X_train = dataset_configurations.get("X_train")
        y_train = dataset_configurations.get("y_train")
        X_test = dataset_configurations.get("X_test")
        y_true = dataset_configurations.get("y_true")
        dataset_metadata = dataset_configurations.get("dataset_metadata")
        feature_metadata = dataset_configurations.get("feature_metadata")
        task_type = dataset_metadata.get("task_type")

        result_row_dict = {
            "embedding_model": None,
            "algorithm": embedding_model.name,
            "fold": feature_metadata["fold"],
            "repeat": feature_metadata["repeat"],
        }

        result_row_dict.update(dataset_metadata)

        test_prediction = embedding_model.get_end_to_end_prediction(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            task_type=task_type,
        )

        metric_scores = self._compute_metrics(
            y_true,
            test_prediction,
            task_type,
        )

        result_row_dict.update(metric_scores)

        self._results_buffer.append(result_row_dict)

    def _get_task_configuration(self, dataset, task) -> tuple[int, int]:
        """Get the number of folds and repeats for a task.

        Args:
            dataset: OpenML dataset object.
            task: OpenML task object.

        Returns:
            Tuple of (n_folds, n_repeats).
        """
        if self.tabarena_lite:
            return 1, 1

        _, folds, _ = task.get_split_dimensions()
        n_samples = dataset.qualities["NumberOfInstances"]

        if n_samples < 2_500:
            tabarena_repeats = 10
        elif n_samples > 250_000:
            tabarena_repeats = 1
        else:
            tabarena_repeats = 3

        return folds, tabarena_repeats

    def _remove_columns_with_one_unique_value(
        self,
        X: pd.DataFrame,
        categorical_indicator: List[bool],
        dataset_name: str = "",
    ) -> tuple[pd.DataFrame, List[bool]]:
        """
        Removes columns with only one unique value from a DataFrame and updates
        the categorical indices accordingly.

        This method processes the input DataFrame and identifies columns with only
        one unique value (ignoring NaN values). These columns are dropped from the DataFrame,
        and the corresponding categorical indices are updated to reflect the changes.

        Args:
            X (pd.DataFrame): The input DataFrame containing data.
            categorical_indicator (List[bool]): A list of boolean values where each entry
                indicates whether the corresponding column in X is categorical.
            dataset_name (str): An optional name for the dataset, used for logging
                purposes. Defaults to an empty string.

        Returns:
            tuple[pd.DataFrame, List[bool]]: A tuple containing the updated DataFrame after
                removing columns with only one unique value and the updated list of
                categorical indices.
        """
        X_copy = X.copy()

        num_features_before = X_copy.shape[1]
        categorical_indices_updated = categorical_indicator.copy()

        # Get column names
        cols = [col for col, is_cat in zip(X.columns, categorical_indicator)]

        # Track columns to drop
        cols_to_drop = []

        # Check each column
        for col in cols:
            n_unique = X_copy[col].nunique(dropna=True)
            if n_unique <= 1:
                cols_to_drop.append(col)

        if len(cols_to_drop) > 0:
            self.logger.info(f"Dataset: {dataset_name}")
            self.logger.info(f"Number of features before: {num_features_before}")
            self.logger.info(
                f"Dropping {len(cols_to_drop)} columns with "
                f"only one distinct value in dataset {dataset_name}."
            )

        # Drop columns with one category
        if cols_to_drop:
            X_copy = X_copy.drop(columns=cols_to_drop)

            # Update categorical_indices by removing entries for dropped columns
            drop_indices = [X.columns.get_loc(col) for col in cols_to_drop]
            categorical_indices_updated = [
                cat_ind
                for i, cat_ind in enumerate(categorical_indicator)
                if i not in drop_indices
            ]
            self.logger.info(f"Number of features after: {X_copy.shape[1]}")
        return X_copy, categorical_indices_updated


def run_tabarena_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    exclude_datasets: list[str] | None = None,
    upper_bound_num_samples: int = 100000,
    upper_bound_num_features: int = 500,
    result_dir: str | Path = "result_tabarena",
    save_result_dataframe: bool = True,
    timestamp: str = TIMESTAMP,
    run_tabpfn_subset: bool = True,
    openml_cache_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Run the TabArena benchmark for a set of embedding models.

    This function evaluates the performance of specified embedding models on a suite
    of tasks from the TabArena benchmark. It computes AUC scores for classification
    and MAPE scores for regression tasks.

    Args:
        embedding_models: List of embedding model instances to evaluate.
        evaluators: List of evaluator instances.
        tabarena_version: The version identifier for the TabArena benchmark study.
            Defaults to "tabarena-v0.1".
        tabarena_lite: Whether to run in lite mode with fewer splits and repetitions.
            Defaults to True.
        exclude_datasets: List of dataset names to exclude from the benchmark.
        upper_bound_num_samples: Maximum dataset size to consider. Datasets larger
            than this will be skipped. Defaults to 100000.
        upper_bound_num_features: Maximum number of features to consider. Datasets
            with more features will be skipped. Defaults to 500.
        result_dir: Directory path for saving results. Defaults to "result_tabarena".
        save_result_dataframe: Whether to save results to disk. Defaults to True.
        timestamp: Timestamp string for result file naming. Defaults to current timestamp.
        run_tabpfn_subset: Whether to run only the TabPFN subset of tasks. Defaults to True.
        openml_cache_dir: Directory for caching OpenML datasets. If None, uses default.

    Returns:
        pl.DataFrame: Polars DataFrame containing the benchmark results.
    """
    benchmark = TabArenaBenchmark(
        tabarena_version=tabarena_version,
        tabarena_lite=tabarena_lite,
        exclude_datasets=exclude_datasets,
        result_dir=result_dir,
        timestamp=timestamp,
        save_result_dataframe=save_result_dataframe,
        upper_bound_num_samples=upper_bound_num_samples,
        upper_bound_num_features=upper_bound_num_features,
        run_tabpfn_subset=run_tabpfn_subset,
        openml_cache_dir=openml_cache_dir,
    )

    return benchmark.run_benchmark(embedding_models, evaluators)
