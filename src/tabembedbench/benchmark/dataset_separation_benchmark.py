import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import openml
import polars as pl
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
from tabembedbench.utils.visualization_utils import create_interactive_embedding_plot_3d


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

    filtered_task_ids = []

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        if not (
            dataset.qualities["NumberOfInstances"] > max_num_samples
            or dataset.qualities["NumberOfFeatures"] > max_num_features
            or dataset.qualities["NumberOfMissingValues"] > 0
        ):
            filtered_task_ids.append(task_id)

    task_ids = filtered_task_ids
    rng = np.random.default_rng(random_seed)

    if len(task_ids) < 2:
        raise ValueError(
            f"Not enough filtered tasks ({len(task_ids)}) to create collections. "
            f"Adjust max_num_samples ({max_num_samples}) or max_num_features ({max_num_features})."
        )

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
) -> Dict[str, str | List[int]]:
    """
    Loads dataset collections from a JSON file and returns them as a dictionary.

    This function reads a JSON file containing dataset collections and parses it into
    a dictionary. The dictionary keys represent the dataset collection names, and the
    values correspond to either string information or a list of integers, depending on
    the content of the JSON file.

    Args:
        json_file_path (str | Path): Path to the JSON file containing dataset collections.

    Returns:
        Dict[str, str | List[int]]: A dictionary with dataset collection names as keys and
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
    """
    Handles the benchmarking process for testing dataset separation of different embedding models, utilizing dataset
    collections from OpenML and applying machine learning tasks for comparative studies.

    This class enables loading, preprocessing, and evaluating collections of datasets
    using user-defined metrics and embedding models. It supports configuration of logging,
    caching, random seed management, and result saving.

    Attributes:
        list_dataset_collections (Dict[str, str | List[int]]): Dictionary containing
            collections of datasets to be used in the benchmarking process, where each
            key represents a collection and the value includes task IDs and additional
            metadata.
        random_seed (Optional[int]): Seed for the random number generator used for
            reproducibility.
        rng (numpy.random.Generator): Instance of the random number generator for
            random operations.
        create_embedding_plots (bool): Indicates whether to create plots of embeddings
            for visualization purposes.
        openml_cache_dir (Path): Directory path where OpenML datasets are cached for reuse.
    """

    def __init__(
        self,
        list_dataset_collections: Dict[str, str | List[int]],
        random_seed: Optional[int] = None,
        result_dir: str | Path = "result_dataset_separation",
        timestamp: str | None = None,
        logging_level: int = logging.INFO,
        save_result_dataframe: bool = True,
        create_embedding_plots: bool = False,
        benchmark_metrics: (
            Dict[str, Dict[str, Callable[[np.ndarray, np.ndarray], float]]] | None
        ) = None,
        openml_cache_dir: str | Path | None = None,
        google_bucket: str | None = None,
    ):
        super().__init__(
            name="TabEmbedBench_DatasetSeparation",
            task_type=CLASSIFICATION_TASKS,
            result_dir=result_dir,
            timestamp=timestamp,
            logging_level=logging_level,
            save_result_dataframe=save_result_dataframe,
            benchmark_metrics=benchmark_metrics,
            gcs_bucket_name=google_bucket,
        )
        self.list_dataset_collections = list_dataset_collections
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.create_embedding_plots = create_embedding_plots

        if openml_cache_dir is None:
            openml_cache_dir = Path("data/tabarena_datasets")
        else:
            openml_cache_dir = Path(openml_cache_dir)

        openml_cache_dir.mkdir(parents=True, exist_ok=True)
        openml.config.set_root_cache_directory(openml_cache_dir)
        self.openml_cache_dir = openml_cache_dir

    def _load_datasets(self, **kwargs) -> Dict[str, Any]:
        """
        Loads dataset collections and their associated tasks. This method retrieves tasks
        based on predefined collections, extracts datasets and their splits, and organizes
        them into a structured format.

        Args:
            **kwargs: Arbitrary keyword arguments. These arguments are not used in the
                current implementation but can be passed for future extensions.

        Returns:
            Dict[str, Any]: A dictionary where the key is the name of each dataset
            collection and the value is a dictionary containing:
                - `name` (str): The name of the dataset collection.
                - `selected_tasks_str` (str): A string containing the selected task IDs
                  for the collection.
                - `collection` (list[dict]): A list of dictionaries. Each dictionary corresponds
                  to a task and includes:
                    - `task_id` (int): The ID of the task.
                    - `task` (openml.tasks.Task): The OpenML task object.
                    - `dataset` (openml.datasets.Dataset): The dataset associated with the task.
                    - `folds` (int): The number of folds in the dataset split.
                    - `label` (int): The index of the task for ordering.
        """
        dataset_collections = {}
        for collection_name, collection in self.list_dataset_collections.items():
            selected_tasks_str = collection["selected_task_ids_str"]
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
        """
        Prepares and yields a dataset collection for processing, including train/test splits,
        label encoding, and metadata.

        Args:
            dataset_collection (Dict[str, str | List[int]]): A dictionary containing information about
                the dataset collection, with attributes such as the name of the collection,
                the selected tasks as a string, and a list of dataset configurations. Each
                dataset configuration includes details such as the associated task, dataset
                object, and label for encoding.

        Yields:
            Iterator[dict]: An iterator that yields dictionaries containing structured
                information about the dataset collection, including train/test splits, label
                encoding, metadata about the dataset, and feature information (such as
                categorical indices and column names).
        """
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
                "dataset_name": dataset_collection["name"],
                "num_features": max_features,
                "num_samples": max_samples,
            },
            "dataset_collection": list_dataset_configurations,
        }

    def _get_default_metrics(
        self,
    ) -> dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]:
        """
        Returns the default metrics configuration for different supervised learning tasks.

        The metrics include pre-configured scoring functions for binary classification
        and multiclass classification tasks. Each metric is represented by a callable
        function, which accepts two numpy arrays as inputs: the true labels and predicted
        labels, respectively.

        Returns:
            dict[str, dict[str, Callable[[np.ndarray, np.ndarray], float]]]: A dictionary
            where keys represent supervised learning task types (e.g., binary or
            multiclass classification), and the values are dictionaries mapping metric
            names to their respective scoring functions.
        """
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Aggregates embeddings and labels from a collection of datasets and returns
        them as shuffled training and testing sets. Embeddings and labels from each
        dataset in the dataset collection are concatenated to create combined sets.

        Args:
            embedding_model: The embedding generator used to encode dataset inputs
                into feature vectors.
            dataset_collection: A collection of dataset configurations, each being
                a dictionary with necessary data and labels for training and testing.

        Returns:
            A tuple containing:
                - numpy.ndarray: Shuffled training embeddings.
                - numpy.ndarray: Shuffled training labels.
                - numpy.ndarray: Combined test embeddings.
                - numpy.ndarray: Combined test labels.
                - dict: Metadata for the process
        """
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

        embedding_metadata = {
            "embedding_model": embedding_model.name,
        }

        return (
            combined_train_embeddings[indices],
            combined_train_labels[indices],
            combined_test_embeddings,
            combined_test_labels,
            embedding_metadata,
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

        (
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
            embedding_metadata,
        ) = self._get_embeddings_from_dataset_collection(
            embedding_model, dataset_collection
        )

        dataset_collection_configuration = {
            "y_train": train_labels,
            "task_type": task_type,
        }

        self._embedding_utils(
            (train_embeddings, test_embeddings, embedding_metadata),
            {},
            dataset_configurations,
        )

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
                (train_embeddings, test_embeddings, embedding_metadata),
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


def run_dataseparation_benchmark(
    embedding_models: List[AbstractEmbeddingGenerator],
    evaluators: List[AbstractEvaluator],
    max_num_samples: int = 100000,
    max_num_features: int = 500,
    num_collections: int = 10,
    use_tabpfn_subset: bool = False,
    result_dir: str | Path = "result_dataset_separation",
    save_result_dataframe: bool = True,
    timestamp: str | None = None,
    create_embedding_plots: bool = False,
    dataset_configurations_json_path: str | Path = None,
    openml_cache_dir: str | Path | None = None,
    google_bucket: str | None = None,
) -> pl.DataFrame:
    if not dataset_configurations_json_path:
        dataset_collections = get_list_of_dataset_collections(
            max_num_samples=max_num_samples,
            max_num_features=max_num_features,
            num_collections=num_collections,
            use_tabpfn_subset=use_tabpfn_subset,
        )
    else:
        dataset_collections = load_dataset_collections_json(
            dataset_configurations_json_path
        )

    benchmark = DatasetSeparationBenchmark(
        list_dataset_collections=dataset_collections,
        result_dir=result_dir,
        save_result_dataframe=save_result_dataframe,
        openml_cache_dir=openml_cache_dir,
        create_embedding_plots=create_embedding_plots,
        timestamp=timestamp,
        google_bucket=google_bucket,
    )

    return benchmark.run_benchmark(embedding_models, evaluators)
