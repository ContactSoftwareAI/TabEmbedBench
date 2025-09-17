from typing import Dict
import numpy as np

def update_result_dict(
    result_dict: Dict[str, list],
    dataset_name: str,
    dataset_size: int,
    embedding_model_name: str,
    num_neighbors: int,
    compute_time: float,
    task: str,
    auc_score: float = None,
    msr_score: float = None,
    distance_metric: str = "euclidean",
    outlier_benchmark: bool = False,
):
    result_dict["dataset_name"].append(dataset_name)
    result_dict["dataset_size"].append(dataset_size)
    result_dict["embedding_model"].append(embedding_model_name)
    result_dict["num_neighbors"].append(num_neighbors)
    result_dict["time_to_compute_embeddings"].append(compute_time)
    result_dict["benchmark"].append("tabarena")
    result_dict["distance_metric"].append(distance_metric)
    result_dict["task"].append(task)

    if auc_score is not None:
        result_dict["auc_score"].append(auc_score)
        if not outlier_benchmark:
            result_dict["msr_score"].append(np.inf)
    else:
        result_dict["auc_score"].append((-1) * np.inf)
        result_dict["msr_score"].append(msr_score)