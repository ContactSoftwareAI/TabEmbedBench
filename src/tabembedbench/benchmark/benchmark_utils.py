from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.dataset_utils import download_adbench_tabular_datasets, get_data_description


def run_outlier_benchmark(
    model: BaseEmbeddingGenerator,
    dataset_paths: Optional[Union[str, Path]] = None,
    random_state: int = 42,
    save_embeddings: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """
    Runs an outlier detection benchmark using a given embedding generator model and
    datasets. The function handles downloading default datasets if none are provided,
    splitting datasets into training and testing sets, computing descriptions of the
    datasets, and running the benchmark experiment.

    Args:
        model (BaseEmbeddingGenerator): The embedding generator model used
            for the benchmark experiment.
        dataset_paths (Optional[Union[str, Path]]): Path to the directory containing
            datasets in .npz format. Defaults to "data/adbench_tabular_datasets".
        random_state (int): The random state value for reproducibility of
            train-test splits. Defaults to 42.
        save_embeddings (bool): Boolean flag indicating whether to save embeddings
            generated during the experiment. Defaults to False.
        save_path (Optional[Union[str, Path]]): Path to save the embeddings if
            `save_embeddings` is True. Defaults to None.

    Returns:
        pl.DataFrame: A DataFrame containing the results of the benchmark experiment.

    Raises:
        ValueError: If any of the dataset files are corrupted or incomplete,
            causing a mismatch in data loading.
    """
    print("Running outlier benchmark...")
    if dataset_paths is None:
        dataset_paths = Path("data/adbench_tabular_datasets")
        if not dataset_paths.exists():
            print("Downloading ADBench tabular datasets...")
            download_adbench_tabular_datasets(dataset_paths)
    else:
        dataset_paths = Path(dataset_paths)
    print(f"Dataset paths: {dataset_paths}")

    npz_files = list(dataset_paths.glob("*.npz"))
    print(f"Found {len(npz_files)} .npz files: {[f.name for f in npz_files]}")

    benchmark_result_df = None

    for dataset_file in dataset_paths.glob("*.npz"):
        print(f"Running benchmark for {dataset_file.name}...")
        dataset = np.load(dataset_file)

        X = dataset["X"]
        y = dataset["y"]

        dataset_description = get_data_description(
            dataset["X"], dataset["y"], dataset_file.stem
        )

        dataset_df = pl.DataFrame(dataset_description)

        result_df = run_experiment(
            model,
            X=X,
            y=y,
            save_embeddings=save_embeddings,
            save_path=save_path,
        )

        dataset_df = dataset_df.join(result_df, how="cross")

        if benchmark_result_df is None:
            benchmark_result_df = dataset_df
        else:
            benchmark_result_df = pl.concat([benchmark_result_df, dataset_df])

        if benchmark_result_df is None:
            raise ValueError("Benchmark result DataFrame is empty.")

    return benchmark_result_df


def run_experiment(
    model: BaseEmbeddingGenerator,
    X: Union[torch.Tensor, np.ndarray],
    y: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_embeddings: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    result_dict = dict()
    result_dict["algorithm"] = []
    result_dict["neighbors"] = []

    num_train_samples = X.shape[0]

    containment_rate = np.sum(y) / num_train_samples

    X_embed = model.compute_embeddings(X)

    num_neighbors_list = [i for i in range(1, 50)]

    for num_neighbors in num_neighbors_list:
        lof = LocalOutlierFactor(
            n_neighbors=num_neighbors,
            n_jobs=-1,
        )
        lof.fit(X_embed)
        y_pred = lof.fit_predict(X_embed)

        score_train = compute_metrics(y, y_pred)

        result_dict["algorithm"].append("lof")
        result_dict["neighbors"].append(num_neighbors)

        for key, item in score_train.items():
            if key in result_dict.keys():
                result_dict[key].append(item)
            else:
                result_dict[key] = [item]

        lof_with_train_containment = LocalOutlierFactor(
            n_neighbors=num_neighbors,
            n_jobs=-1,
            contamination=containment_rate,
        )

        lof_with_train_containment.fit(X_embed)
        y_pred = lof_with_train_containment.fit_predict(X_embed)
        score_train = compute_metrics(y, y_pred)

        result_dict["algorithm"].append("lof_with_train_containment")
        result_dict["neighbors"].append(num_neighbors)
        for key, item in score_train.items():
            result_dict[key].append(item)

    result_df = pl.DataFrame(result_dict)

    return result_df


def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics based on provided true labels and predicted labels.

    Args:
        y_true: List or array of true binary labels (0 or 1).
        y_pred: List or array of predicted probabilities or scores.

    Returns:
        dict: A dictionary containing the computed AUC score under the key
        'auc_score'.
    """
    return {"auc_score": roc_auc_score(y_true, y_pred)}