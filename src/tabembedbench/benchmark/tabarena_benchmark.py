import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import openml
import pandas as pd
import polars as pl
from sklearn.metrics import log_loss, mean_absolute_percentage_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.embedding_models.abstractembedding import (
    AbstractEmbeddingGenerator,
)
from tabembedbench.evaluators.abstractevaluator import AbstractEvaluator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

SUPERVISED_REGRESSION = "Supervised Regression"
SUPERVISED_BINARY_CLASSIFICATION = "Supervised Binary Classification"
SUPERVISED_MULTICLASSIFICATION = "Supervised Multiclass Classification"

TABARENA_TABPFN_SUBSET = [
    363621,
    363629,
    363614,
    363698,
    363626,
    363685,
    363625,
    363696,
    363675,
    363707,
    363671,
    363612,
    363615,
    363711,
    363682,
    363684,
    363674,
    363700,
    363702,
    363704,
    363623,
    363694,
    363708,
    363706,
    363689,
    363624,
    363619,
    363676,
    363712,
    363632,
    363691,
    363681,
    363686,
    363679,
]

TASK_IDS_WITH_MISSING_VALUES = [363671, 363679, 363684, 363694, 363711, 363712]


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
        """Initialize the TabArena benchmark.

        Args:
            tabarena_version: OpenML suite identifier.
            tabarena_lite: Whether to use lite mode (fewer folds/repeats).
            exclude_datasets: List of dataset names to exclude from the benchmark.
            result_dir: Directory for saving results.
            timestamp: Timestamp string for result file naming.
            save_result_dataframe: Whether to save results to disk.
            upper_bound_num_samples: Maximum dataset size to process.
            upper_bound_num_features: Maximum number of features to process.
            run_tabpfn_subset: Whether to run only a subset of TabPFN tasks.
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

        # Setup OpenML cache (similar to OutlierBenchmark's dataset_paths)
        if openml_cache_dir is None:
            openml_cache_dir = Path("data/tabarena_datasets")
        else:
            openml_cache_dir = Path(openml_cache_dir)

        openml_cache_dir.mkdir(parents=True, exist_ok=True)
        openml.config.set_root_cache_directory(openml_cache_dir)
        self.openml_cache_dir = openml_cache_dir

        self.logger.info(f"OpenML cache directory: {self.openml_cache_dir}")

    def _load_datasets(self, **kwargs) -> List[Dict[str, Any]]:
        """Load TabArena tasks from OpenML.

        Returns:
            List of dictionaries containing task information.
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
            return True, f"Skipping dataset {dataset.name}: {reason}"

        self.len_tabpfn_subset -= 1
        task = dataset_info["task"]
        msg = (
            f"Starting experiments for dataset {dataset.name} "
            f"and task {task.task_type}. "
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
                "mape_score": mean_absolute_percentage_error,
            },
            SUPERVISED_BINARY_CLASSIFICATION: {
                "auc_score": roc_auc_score,
            },
            SUPERVISED_MULTICLASSIFICATION: {
                "auc_score": partial(roc_auc_score, multi_class="ovr"),
                "log_loss_score": log_loss,
            },
        }

    def _get_evaluator_prediction(
        self,
        embeddings: Tuple[np.ndarray, np.ndarray, float],
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> dict:
        """Evaluate embeddings for classification or regression.

        Args:
            embeddings: Tuple of (train_embeddings, test_embeddings, compute_time).
            evaluator: The evaluator to use.
            dataset_configurations: Dictionary with data and metadata.

        Returns:
            Dictionary containing evaluation results.
        """
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

        return test_prediction

    def _process_end_to_end_model_pipeline(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        data_split: dict,
    ) -> dict:
        X_train = data_split.get("X_train")
        y_train = data_split.get("y_train")
        X_test = data_split.get("X_test")
        y_test = data_split.get("y_test")
        metadata = data_split.get("metadata")
        task_type = metadata.get("task_type")

        _, X_test_preprocessed = embedding_model.preprocess_data(
            X_train=X_train, X_test=X_test, y_train=y_train, outlier=False, **metadata
        )

        test_prediction = embedding_model.get_prediction(
            X=X_test_preprocessed,
        )

        # Build result dictionary
        result_dict = {
            "dataset_name": [data_split["dataset_name"]],
            "dataset_size": [data_split["dataset_size"]],
            "num_features": [data_split["num_features"]],
            "algorithm": [embedding_model.name],
            "fold": [data_split["metadata"]["fold"]],
            "repeat": [data_split["metadata"]["repeat"]],
        }

        result_dict = self._compute_metrics(
            result_dict,
            y_test,
            test_prediction,
            task_type,
        )

        return result_dict

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


if __name__ == "__main__":
    from tabembedbench.embedding_models import TableVectorizerEmbedding
    from tabembedbench.evaluators import KNNClassifierEvaluator, KNNRegressorEvaluator

    evaluators = [
        KNNRegressorEvaluator(num_neighbors=1, metric="euclidean", weights="distance"),
        KNNClassifierEvaluator(num_neighbors=1, metric="euclidean", weights="distance"),
    ]

    embedding_models = [TableVectorizerEmbedding()]

    result = run_tabarena_benchmark(
        embedding_models,
        evaluators,
        upper_bound_num_samples=1000,
        upper_bound_num_features=100,
    )

    print(result)
