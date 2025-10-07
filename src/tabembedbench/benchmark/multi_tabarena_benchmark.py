from datetime import datetime
from pathlib import Path

import numpy as np
import openml
import polars as pl
from sklearn.metrics import mean_absolute_percentage_error, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from tabicl.sklearn.preprocessing import TransformToNumerical

from tabembedbench.benchmark import AbstractBenchmark
from tabembedbench.evaluators import AbstractEvaluator


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class MultiTabArenaBenchmark(AbstractBenchmark):
    def __init__(
            self,
            tabarena_version: str = "tabarena-v0.1",
            tabarena_lite: bool = True,
            result_dir: str | Path = "result_tabarena",
            timestamp: str = TIMESTAMP,
            save_result_dataframe: bool = True,
            upper_bound_num_samples: int = 100000,
            upper_bound_num_features: int = 500,
            max_number_of_dataset: int = 5,
            use_all_datasets: bool = True,
            random_seed: int = 42
    ):
        super().__init__(
            logger_name="TabEmbedBench_MultiTabArena",
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
        self.max_number_of_dataset = max_number_of_dataset,
        self.use_all_datasets = use_all_datasets
        self.random_seed = random_seed

        self.rng = np.random.default_rng(self.random_seed)

    def _load_datasets(self, **kwargs):
        self.benchmark_suite = openml.study.get_suite(self.tabarena_version)
        self.task_ids = self.benchmark_suite.tasks

        binary_classification_datasets = []
        multiclass_datasets = []

        for task_id in self.task_ids:
            task = openml.tasks.get_task(task_id)
            if task.task_type == "Supervised Classification":
                dataset = task.get_dataset()
                if dataset.qualities["NumberOfInstances"] > self.upper_bound_num_samples:
                    continue
                if dataset.qualities["NumberOfFeatures"] > self.upper_bound_num_features:
                    continue

                num_class = len(dataset.retrieve_class_labels())

                if num_class > 2:
                    multiclass_datasets.append(task_id)
                else:
                    binary_classification_datasets.append(task_id)
            else:
                continue

        binary_batches = self._get_dataset_choices(binary_classification_datasets)
        multiclass_batches = self._get_dataset_choices(multiclass_datasets)

    def _get_dataset_choices(self, classification_ids):
        ids_array = np.array(classification_ids)
        self.rng.shuffle(ids_array)

        batches = []
        start = 0
        while start < len(ids_array):
            remaining = len(ids_array) - start
            if remaining > 1:
                size = self.rng.integers(
                    2, min(self.max_number_of_dataset, remaining)+1
                )
                batches.append(ids_array[start:start+size])
            else:
                size = 1
                batches.append(ids_array[start:start+1])
            start += size

        return batches

    def _get_batches_metadata(self, batches: list[int]):
        pass

    def _should_skip_dataset(self, dataset, **kwargs) -> tuple[
        bool, str | None]:
        pass

    def _prepare_data(self, dataset_info, **kwargs):
        tasks = dataset_info["tasks"]

        label_offset = 0
        for task_id in tasks:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()

            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=task.target_name, dataset_format="dataframe"
            )

            categorical_indices = np.nonzero(categorical_indicator)[0]
            categorical_indices = categorical_indices.tolist()

            train_indices, test_indices = task.get_train_test_split_indices(
                fold=1,
                repeat=1
            )

            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_test = y.iloc[test_indices]

            # Preprocess data
            numerical_transformer = TransformToNumerical()
            X_train = numerical_transformer.fit_transform(X_train)
            X_test = numerical_transformer.transform(X_test)

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)

            max_label = np.max(y_train)
            y_train += label_offset
            y_test += label_offset
            label_offset += max_label + 1


    def _evaluate_embeddings(
        self, embeddings, evaluator: AbstractEvaluator, dataset_info: dict, **kwargs
    ) -> dict:
        tasks = dataset_info["tasks"]
        for task_id in tasks:
            pass

    def _get_benchmark_name(self) -> str:
        pass

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