import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import openml
from pydantic import BaseModel, RootModel, ValidationError
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
)

from tabembedbench.benchmark.abstract_benchmark import AbstractBenchmark
from tabembedbench.constants import (
    CLASSIFICATION_TASKS,
    SUPERVISED_BINARY_CLASSIFICATION,
    SUPERVISED_MULTICLASSIFICATION,
    TABARENA_TABPFN_SUBSET,
)
from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.evaluators import AbstractEvaluator


class DatasetCollection(BaseModel):
    selected_task_ids_str: str
    selected_task_ids: List[int]


class DatasetCollections(RootModel):
    root: Dict[str, DatasetCollection]


def get_list_of_dataset_collections(
    max_num_samples: int,
    max_num_features: int,
    max_num_per_collections: int = 10,
    num_collections: int = 20,
    tabarena_version: str = "tabarena-v0.1",
    use_tabpfn_subset: bool = False,
    random_seed: Optional[int] = None,
) -> Dict[str, str | List[int]]:
    """
    Generates a list of dataset collections based on specified constraints.

    This function filters and groups dataset tasks from OpenML to create collections
    that adhere to the provided constraints on the number of samples and features.
    Each collection contains a group of task identifiers selected randomly.

    Args:
        max_num_samples (int): The maximum number of samples allowed for the dataset tasks
            in the collections.
        max_num_features (int): The maximum number of features allowed for the dataset tasks
            in the collections.
        max_num_per_collections (int, optional): The maximum number of dataset tasks to be
            included in a single collection. Defaults to 10.
        num_collections (int, optional): The total number of collections to generate.
            Defaults to 20.
        tabarena_version (str, optional): The version of the TabArena suite to be used for
            obtaining dataset tasks. Defaults to "tabarena-v0.1".
        use_tabpfn_subset (bool, optional): Whether to use a predefined subset of task IDs
            labeled as TABARENA_TABPFN_SUBSET. Defaults to False.
        random_seed (Optional[int], optional): The random seed for reproducibility of
            the random task selection process. Defaults to None.

    Returns:
        Dict[str, str | List[int]]: A dictionary with dataset collection names as keys and each item
            represents a dataset collection with the following keys:
            - "selected_task_ids_str" (str): A string representation of the selected task IDs.
            - "selected_task_ids" (List[int]): A list of selected task IDs in the collection.
    """
    task_ids = (
        TABARENA_TABPFN_SUBSET
        if use_tabpfn_subset
        else openml.study.get_suite(tabarena_version).tasks
    )

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if (
            dataset.qualities["NumberOfInstances"] > max_num_samples
            or dataset.qualities["NumberOfFeatures"] > max_num_features
        ):
            task_ids.remove(task_id)

    rng = np.random.default_rng(random_seed)

    collections = {}

    for idx in range(num_collections):
        collection_size = rng.integers(2, max_num_per_collections)

        selected_task_ids = rng.choice(
            task_ids,
            size=collection_size,
            replace=False,
        ).tolist()

        selected_task_ids_str = "_".join(
            [str(task_id) for task_id in selected_task_ids]
        )

        name = "collection_" + str(idx)

        collections[name] = {
            "selected_task_ids_str": selected_task_ids_str,
            "selected_task_ids": list(selected_task_ids),
        }

    return collections


def save_dataset_collections(
    dataset_collections: Dict[str, str | List[int]],
    output_file_name: str,
    output_dir: str | Path,
) -> None:
    """
    Saves the given dataset collections to a JSON file in the specified directory.

    This function serializes the dataset collections into a JSON-formatted file
    and saves it with the specified file name in the provided output directory.

    Args:
        dataset_collections (Dict[str, str | List[int]]): Dictionary containing dataset keys and their
            corresponding values, which can be strings or lists of integers.
        output_file_name (str): The name of the output file (without the extension) where the data
            will be saved.
        output_dir (str | Path): The directory path where the JSON file should be saved.

    """
    output_path = (Path(output_dir) / output_file_name).with_suffix(".json")

    with open(str(output_path), "w") as f:
        json.dump(dataset_collections, f)


def load_dataset_collections_json(
    json_file_path: str | Path,
) -> DatasetCollections:
    """
    Loads dataset collections from a JSON file and returns them as a dictionary.

    This function reads a JSON file containing dataset collections and parses it into
    a dictionary. The dictionary keys represent the dataset collection names, and the
    values correspond to either string information or a list of integers, depending on
    the content of the JSON file.

    Args:
        json_file_path (str | Path): Path to the JSON file containing dataset collections.

    Returns:
        DatasetCollections: A dictionary with dataset collection names as keys and
        their corresponding data (either a string or a list of integers) as values.
    """
    json_file_path = Path(json_file_path)

    if not json_file_path.exists():
        raise FileNotFoundError(f"The file {json_file_path} does not exist.")

    try:
        with open(str(json_file_path), "r") as f:
            data = json.load(f)

        dataset_collections = DatasetCollections.model_validate(data)

        return dataset_collections.model_dump()

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_file_path}: {e}")
    except ValidationError as e:
        raise ValueError(f"Schema validation failed for {json_file_path}:\n{e}")


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
            "dataset_collection": list_dataset_configurations,
        }

    def _get_default_metrics(
        self,
    ) -> dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        return {
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

    def _get_embeddings_from_dataset_collection(
        self,
        embedding_model: AbstractEmbeddingGenerator,
        dataset_collection: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_embeddings = []
        test_embeddings = []
        train_labels = []
        test_labels = []

        for dataset_config in dataset_collection:
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
        dataset_configurations: list[dict],
    ) -> None:
        dataset_collection = dataset_configurations["dataset_collection"]

        logger_prefix = f"Collection: {dataset_configurations['name']} - Embedding Model: {embedding_model.name}"

        task_type = (
            SUPERVISED_MULTICLASSIFICATION
            if len(dataset_collection) > 2
            else SUPERVISED_BINARY_CLASSIFICATION
        )

        train_embeddings, train_labels, test_embeddings, test_labels = (
            self._get_embeddings_from_dataset_collection(
                embedding_model, dataset_collection
            )
        )

        dataset_collection_configuration = {
            "y_train": train_labels,
            "task_type": task_type,
        }

        result_row_dict = {
            "collection": dataset_configurations["name"],
            "task_ids": dataset_configurations["selected_tasks_str"],
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
