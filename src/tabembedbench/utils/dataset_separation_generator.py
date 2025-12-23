from typing import Any, Dict, List, Optional

import numpy as np
import openml

from tabembedbench.constants import TABARENA_TABPFN_SUBSET


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
        List[Dict[str, str | List[int]]]: A list of dictionaries, where each dictionary
            represents a dataset collection with the following keys:
            - "name" (str): The name of the collection.
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
