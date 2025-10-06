from datetime import datetime
from pathlib import Path

import numpy as np
import openml
import polars as pl
from sklearn.metrics import (
    mean_absolute_percentage_error,
    roc_auc_score,
    log_loss
)
from sklearn.preprocessing import LabelEncoder
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator
from tabembedbench.evaluators.abstractevaluator import AbstractEvaluator

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class TabArenaBenchmark(AbstractBenchmark):
    """Benchmark for TabArena classification and regression tasks.

    This benchmark evaluates embedding models on supervised learning tasks
    from the OpenML TabArena benchmark suite.
    """

    def __init__(
        self,
        tabarena_version: str = "tabarena-v0.1",
        tabarena_lite: bool = True,
        result_dir: str | Path = "result_tabarena",
        timestamp: str = TIMESTAMP,
        save_result_dataframe: bool = True,
        upper_bound_num_samples: int = 100000,
        upper_bound_num_features: int = 500,
    ):
        """Initialize the TabArena benchmark.

        Args:
            tabarena_version: OpenML suite identifier.
            tabarena_lite: Whether to use lite mode (fewer folds/repeats).
            result_dir: Directory for saving results.
            timestamp: Timestamp string for result file naming.
            save_result_dataframe: Whether to save results to disk.
            upper_bound_num_samples: Maximum dataset size to process.
            upper_bound_num_features: Maximum number of features to process.
        """
        super().__init__(
            logger_name="TabEmbedBench_TabArena",
            result_dir=result_dir,
            timestamp=timestamp,
            save_result_dataframe=save_result_dataframe,
            upper_bound_num_samples=upper_bound_num_samples,
            upper_bound_num_features=upper_bound_num_features,
        )

        self.tabarena_version = tabarena_version
        self.tabarena_lite = tabarena_lite
        self.benchmark_suite = None
        self.task_ids = None

    def _load_datasets(self, **kwargs):
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

            datasets.append({
                "task_id": task_id,
                "task": task,
                "dataset": dataset,
                "folds": folds,
                "repeats": repeats,
            })
        
        return datasets

    def _should_skip_dataset(self, dataset_info, **kwargs) -> tuple[bool, str | None]:
        """Check if a dataset should be skipped.

        Args:
            dataset_info: Dictionary containing task and dataset information.
            **kwargs: Additional parameters (unused).

        Returns:
            Tuple of (should_skip, reason).
        """
        dataset = dataset_info["dataset"]
        num_samples = dataset.qualities["NumberOfInstances"]
        num_features = dataset.qualities["NumberOfFeatures"]

        # Check size constraints
        should_skip, reason = self._check_dataset_size_constraints(
            num_samples, num_features, dataset.name
        )

        if not should_skip:
            task = dataset_info["task"]
            self.logger.info(
                f"Starting experiments for dataset {dataset.name} "
                f"and task {task.task_type}"
            )

        return should_skip, reason

    def _prepare_data(self, dataset_info, **kwargs):
        """Prepare data from a TabArena task.

        This method handles the cross-validation splits and data preprocessing
        for TabArena tasks.

        Args:
            dataset_info: Dictionary containing task and dataset information.
            **kwargs: Additional parameters (unused).

        Returns:
            Generator yielding prepared data for each fold/repeat combination.
        """
        task = dataset_info["task"]
        dataset = dataset_info["dataset"]
        folds = dataset_info["folds"]
        repeats = dataset_info["repeats"]

        # Iterate through all folds and repeats
        for repeat in range(repeats):
            for fold in range(folds):
                # Get data
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=task.target_name, dataset_format="dataframe"
                )

                categorical_indices = np.nonzero(categorical_indicator)[0]
                categorical_indices = categorical_indices.tolist()

                # Get train/test split
                train_indices, test_indices = task.get_train_test_split_indices(
                    fold=fold,
                    repeat=repeat,
                )

                X_train = X.iloc[train_indices]
                y_train = y.iloc[train_indices]
                X_test = X.iloc[test_indices]
                y_test = y.iloc[test_indices]

                # Preprocess data
                numerical_transformer = TransformToNumerical()
                X_train = numerical_transformer.fit_transform(X_train)
                X_test = numerical_transformer.transform(X_test)

                # Encode labels for classification
                if task.task_type == "Supervised Classification":
                    label_encoder = LabelEncoder()
                    y_train = label_encoder.fit_transform(y_train)
                    y_test = label_encoder.transform(y_test)

                yield {
                    "data": X_train,
                    "dataset_name": dataset.name,
                    "dataset_size": X.shape[0],
                    "num_features": X.shape[1],
                    "task_type": task.task_type,
                    "embedding_kwargs": {
                        "X_test": X_test,
                        "categorical_indices": categorical_indices,
                    },
                    "eval_kwargs": {
                        "y_train": y_train,
                        "y_test": y_test,
                        "task_type": task.task_type,
                    },
                }

    def _evaluate_embeddings(
        self,
        embedding_results,
        evaluator: AbstractEvaluator,
        dataset_info: dict,
        **kwargs
    ) -> dict:
        """Evaluate embeddings for classification or regression.

        Args:
            embedding_results: Tuple of (train_embeddings, compute_time, test_embeddings, test_compute_time).
            evaluator: The evaluator to use.
            dataset_info: Dictionary with dataset metadata.
            **kwargs: Additional parameters including 'y_train', 'y_test', and 'task_type'.

        Returns:
            Dictionary containing evaluation results.
        """
        train_embeddings = embedding_results[0]
        test_embeddings = embedding_results[2]
        y_train = kwargs.get("y_train")
        y_test = kwargs.get("y_test")
        task_type = kwargs.get("task_type")

        # Train evaluator
        prediction_train, _ = evaluator.get_prediction(
            train_embeddings,
            y_train,
            train=True,
        )

        # Get test predictions
        test_prediction, _ = evaluator.get_prediction(
            test_embeddings,
            train=False,
        )

        # Build result dictionary
        result_dict = {
            **dataset_info,
        }

        # Compute task-specific metrics
        if task_type == "Supervised Regression":
            mape_score = mean_absolute_percentage_error(y_test, test_prediction)
            result_dict["task"] = ["regression"]
            result_dict["mape_score"] = [mape_score]

        elif task_type == "Supervised Classification":
            n_classes = test_prediction.shape[1]
            if n_classes == 2:
                auc_score = roc_auc_score(y_test, test_prediction[:, 1])
                result_dict["task"] = ["classification"]
                result_dict["classification_type"] = ["binary"]
            else:
                auc_score = roc_auc_score(
                    y_test, test_prediction, multi_class="ovr"
                )
                log_loss_score = log_loss(y_test, test_prediction)
                result_dict["task"] = ["classification"]
                result_dict["classification_type"] = ["multiclass"]
                result_dict["log_loss_score"] = [log_loss_score]
            result_dict["auc_score"] = [auc_score]

        return result_dict

    def _get_benchmark_name(self) -> str:
        """Get the benchmark name for result saving.

        Returns:
            String identifier for the benchmark.
        """
        return "TabArena"

    def _is_evaluator_compatible(self, evaluator: AbstractEvaluator, **kwargs) -> bool:
        """Check if evaluator is compatible with the current task.

        Args:
            evaluator: The evaluator to check.
            **kwargs: Additional parameters (unused).

        Returns:
            True if evaluator supports the task type.
        """
        # Get task type from prepared data
        task_type = kwargs.get("task_type")
        if task_type is None:
            return False
        return task_type == evaluator.task_type

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


def run_tabarena_benchmark(
    embedding_models: list[AbstractEmbeddingGenerator],
    evaluators: list[AbstractEvaluator],
    tabarena_version: str = "tabarena-v0.1",
    tabarena_lite: bool = True,
    upper_bound_num_samples: int = 100000,
    upper_bound_num_features: int = 500,
    result_dir: str | Path = "result_tabarena",
    save_result_dataframe: bool = True,
    timestamp: str = TIMESTAMP,
) -> pl.DataFrame:
    """Run the TabArena benchmark for a set of embedding models.

    This function evaluates the performance of specified embedding models on a suite
    of tasks from the TabArena benchmark. Depending on the size of datasets and
    parameters provided, the function dynamically adjusts the number of repetitions
    and splits for the benchmark. It evaluates each embedding model on each task
    and computes the AUC/MSR scores for supervised classification and regression tasks
    with k-nearest neighbors.

    Args:
        embedding_models: List of embedding model instances that inherit from
            BaseEmbeddingGenerator. Each model will be evaluated for its performance.
        evaluators: List of evaluators
        tabarena_version: The version identifier for the TabArena benchmark study.
            Defaults to "tabarena-v0.1".
        tabarena_lite: Boolean indicating whether to run in lite mode. If True, uses fewer
            splits and repetitions for quicker evaluations. Defaults to True.
        upper_bound_num_samples: Integer representing the maximum dataset size to
            consider for benchmarking. Datasets larger than this value will be skipped.
            Defaults to 100000.
        upper_bound_num_features: Integer representing the maximum number of features
            considered for benchmarking. Datasets with more features than this value
            will be skipped. Defaults to 500.
        result_dir: Directory path for saving results.
        save_result_dataframe: Whether to save results to disk.
        timestamp: Timestamp string for result file naming.

    Returns:
        polars.DataFrame: A dataframe summarizing the benchmark results. The columns
            include dataset information, embedding model names, number of neighbors,
            metrics such as AUC/MSR scores, embedding computation time, and benchmark
            type.
    """
    benchmark = TabArenaBenchmark(
        tabarena_version=tabarena_version,
        tabarena_lite=tabarena_lite,
        result_dir=result_dir,
        timestamp=timestamp,
        save_result_dataframe=save_result_dataframe,
        upper_bound_num_samples=upper_bound_num_samples,
        upper_bound_num_features=upper_bound_num_features,
    )

    return benchmark.run_benchmark(embedding_models, evaluators)
