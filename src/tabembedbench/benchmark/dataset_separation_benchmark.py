from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import openml
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.constants import (
    CLASSIFICATION_TASKS,
    SUPERVISED_BINARY_CLASSIFICATION,
    SUPERVISED_MULTICLASSIFICATION,
)
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator


class DatasetSeparationBenchmark(AbstractBenchmark):
    def __init__(
        self,
        list_dataset_collections: Dict[str, str | List[int]],
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            name="Dataset Separation Benchmark",
            task_type=CLASSIFICATION_TASKS,
        )
        self.list_dataset_collections = list_dataset_collections
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def _load_datasets(self, **kwargs) -> Dict[str, Any]:
        dataset_collections = {}
        for collection_name, collection in self.list_dataset_collections.items():
            selected_tasks_str = collection["selected_tasks_str"]
            collection_list = []
            for index, task_id in enumerate(collection["selected_task_ids"]):
                task = openml.tasks.get_task(task_id)
                dataset = task.get_dataset()
                _, folds, _ = task.get_split_dimensions()

                collection_list.append(
                    {
                        "task_id": task_id,
                        "task": task,
                        "dataset": dataset,
                        "folds": folds,
                        "label": index,
                    }
                )
            dataset_collections[collection_name] = {
                "name": collection_name,
                "selected_tasks_str": selected_tasks_str,
                "collection": collection_list,
            }

        return dataset_collections

    def _should_skip_dataset(self, dataset, **kwargs) -> tuple[bool, str]:
        return False, ""

    def _prepare_dataset(
        self, dataset_collection: Dict[str, str | List[int]]
    ) -> Iterator[dict]:
        list_dataset_configurations = []

        max_features = 0
        max_samples = 0

        # TODO: iterate over more folds if interested.
        for dataset_info in dataset_collection["collection"]:
            task = dataset_info["task"]
            dataset = dataset_info["dataset"]
            label = dataset_info["label"]

            dataset_metadata = {
                "dataset_name": dataset.name,
                "num_samples": dataset.qualities.get("NumberOfInstances"),
                "num_features": dataset.qualities.get("NumberOfFeatures"),
            }

            X, _, categorical_indicator, _ = dataset.get_data(
                target=task.target_name, dataset_format="dataframe"
            )

            train_indices, test_indices = task.get_train_test_split_indices(
                fold=0,
                repeat=0,
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
            y_train = label * np.ones(len(train_indices))
            y_test = label * np.ones(len(test_indices))

            max_features = max(max_features, dataset_metadata["num_features"])
            max_samples = max(max_samples, dataset_metadata["num_samples"])

            list_dataset_configurations.append(
                {
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
            )

        yield {
            "name": dataset_collection["name"],
            "selected_tasks_str": dataset_collection["selected_tasks_str"],
            "dataset_metadata": {
                "num_features": max_features,
                "num_samples": max_samples,
            },
            "dataset_collections": list_dataset_configurations,
        }

    def _get_default_metrics(
        self,
    ) -> dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        return {
            SUPERVISED_MULTICLASSIFICATION: {
                "auc_score_ovr": partial(roc_auc_score, multi_class="ovr"),
                "auc_score_ovo": partial(roc_auc_score, multi_class="ovo"),
                "log_loss_score": log_loss,
            },
        }

    def _get_evaluator_prediction(
        self,
        embeddings: tuple[np.ndarray, np.ndarray, float],
        evaluator: AbstractEvaluator,
        dataset_configurations: dict,
    ) -> np.ndarray:
        train_embeddings, test_embeddings, _ = embeddings
        y_train = dataset_configurations["y_train"]
        task_type = dataset_configurations["task_type"]

        evaluator.get_prediction(
            train_embeddings,
            y_train,
            train=True,
        )

        test_prediction, _ = evaluator.get_prediction(
            test_embeddings,
            train=False,
        )

        if task_type == SUPERVISED_BINARY_CLASSIFICATION:
            test_prediction = test_prediction[:, 1]

        return test_prediction

    def _prepare_dataset_collection(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        dataset_collections: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_embeddings = []
        test_embeddings = []
        train_labels = []
        test_labels = []

        for dataset_config in dataset_collections:
            # Generate embeddings for the current dataset in the collection
            train_emb, test_emb, _ = self._generate_embeddings(
                embedding_model, dataset_config
            )

            train_embeddings.append(train_emb)
            test_embeddings.append(test_emb)
            train_labels.append(dataset_config["y_train"])
            test_labels.append(dataset_config["y_true"])

        # Combine all datasets in the collection into one large pool
        combined_train_embeddings = np.concatenate(train_embeddings, axis=0)
        combined_train_labels = np.concatenate(train_labels, axis=0)

        combined_test_embeddings = np.concatenate(test_embeddings, axis=0)
        combined_test_labels = np.concatenate(test_labels, axis=0)

        # Create shuffled indices to keep embeddings and labels synchronized
        indices = np.arange(len(combined_train_embeddings))
        self.rng.shuffle(indices)

        return (
            combined_train_embeddings[indices],
            combined_train_labels[indices],
            combined_test_embeddings,
            combined_test_labels,
        )

    def _process_embedding_model_pipeline(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        evaluators: list[AbstractEvaluator],
        dataset_configurations: len[dict],
    ) -> None:
        dataset_collection_name = dataset_configurations["name"]
        dataset_collections = dataset_configurations["dataset_collections"]

        logger_prefix = f"Collection: {dataset_collection_name} - Embedding Model: {embedding_model.name}"

        task_type = (
            SUPERVISED_MULTICLASSIFICATION
            if len(dataset_collections) > 2
            else SUPERVISED_BINARY_CLASSIFICATION
        )

        train_embeddings, train_labels, test_embeddings, test_labels = (
            self._prepare_dataset_collection(embedding_model, dataset_collections)
        )

        dataset_collection_configuration = {
            "y_train": train_labels,
            "task_type": task_type,
        }

        result_row_dict = {
            "collection": dataset_collection_name,
        }

        for evaluator in evaluators:
            if not self._is_compatible(evaluator, task_type):
                self.logger.debug(
                    f"{logger_prefix} - Skipping evaluator {evaluator.name} is not compatible with {self.task_type}. Skipping..."
                )
                continue
            self.logger.info(
                f"{logger_prefix} - Evaluating embeddings with {evaluator.name}..."
            )
            prediction = self._get_evaluator_prediction(
                (train_embeddings, test_embeddings, 0.0),
                evaluator=evaluator,
                dataset_configurations=dataset_collection_configuration,
            )
            metric_scores = self._compute_metrics(
                y_true=test_labels,
                y_pred=prediction,
                task_type=task_type,
            )
            result_row_dict["algorithm"] = evaluator.name
            result_row_dict.update(metric_scores)
            result_row_dict.update(evaluator.get_parameters())

            evaluator.reset_evaluator()
            self._cleanup_gpu_cache()
            self._results_buffer.append(result_row_dict)

            # Save intermediate results after each model
            self._save_results()
            self._cleanup_gpu_cache()
