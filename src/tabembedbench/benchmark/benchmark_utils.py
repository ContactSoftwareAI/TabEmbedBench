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
    dataset_paths: Optional[str, Path] = None,
    random_state: int = 42,
    save_embeddings: bool = False,
    save_path: Optional[str, Path] = None,
) -> pl.DataFrame:
    if dataset_paths is None:
        dataset_paths = Path("data/adbench_tabular_datasets")
        if not dataset_paths.exists():
            download_adbench_tabular_datasets(dataset_paths)
    else:
        dataset_paths = Path(dataset_paths)

    for dataset_file in dataset_paths.glob("*.npz"):
        print(f"Running benchmark for {dataset_file.name}...")
        dataset = np.load(dataset_file)

        X = dataset["X"]
        y = dataset["y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        dataset_description = get_data_description(
            dataset["X"], dataset["y"], dataset_file.stem
        )

        dataset_df = pl.DataFrame(dataset_description)

        result_df = run_experiment(
            model,
            X_train,
            X_test,
            save_embeddings=save_embeddings,
            save_path=save_path,
        )


def run_experiment(
    model: BaseEmbeddingGenerator,
    X_train: Union[torch.Tensor, np.ndarray],
    X_test: Union[torch.Tensor, np.ndarray],
    y_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
    y_test: Optional[Union[torch.Tensor, np.ndarray]] = None,
    save_embeddings: bool = False,
    save_path: Optional[str, Path] = None,
) -> pl.DataFrame:
    result_dict = dict()
    result_dict["algorithm"] = []
    result_dict["neighbors"] = []
    result_dict["train_score"] = []
    result_dict["test_score"] = []

    num_train_samples = X_train.shape[0]

    containment_rate = np.sum(y_train) / num_train_samples

    X_train_embed = model.compute_embeddings(X_train)
    X_test_embed = model.compute_embeddings(X_test)

    num_neighbors_list = (
        num_train_samples*np.array([0.2, 0.02, 0.01, 0.005, 0.001])
    ).astype(np.int32)

    for num_neighbors in num_neighbors_list:
        lof = LocalOutlierFactor(
            n_neighbors=num_neighbors,
            n_jobs=-1,
        )
        lof.fit(X_train_embed)
        y_pred_train = lof.predict(X_train_embed)

        score_train = compute_metrics(y_train, y_pred_train)

        y_pred_test = lof.predict(X_test_embed)

        score_test = compute_metrics(y_test, y_pred_test)

        result_dict["algorithm"].append("lof")
        result_dict["neighbors"].append(num_neighbors)
        result_dict["train_score"].append(score_train["auc_score"])
        result_dict["test_score"].append(score_test["auc_score"])

        lof_with_train_containment = LocalOutlierFactor(
            n_neighbors=num_neighbors,
            n_jobs=-1,
            contamination=containment_rate,
        )

        lof_with_train_containment.fit(X_train_embed)
        y_pred_train = lof_with_train_containment.predict(X_train_embed)
        score_train = compute_metrics(y_train, y_pred_train)
        y_pred_test = lof_with_train_containment.predict(X_test_embed)
        score_test = compute_metrics(y_test, y_pred_test)

        result_dict["algorithm"].append("lof_with_train_containment")
        result_dict["neighbors"].append(num_neighbors)
        result_dict["train_score"].append(score_train["auc_score"])
        result_dict["test_score"].append(score_test["auc_score"])

    result_df = pl.DataFrame(result_dict)

    top_5_train_results = result_df.sort("train_score", descending=True).head(5)
    top_5_test_results = result_df.sort("test_score", descending=True).head(5)

    print(top_5_train_results)
    print(top_5_test_results)

    return pl.DataFrame(result_dict)


def compute_metrics(y_true, y_pred):
    return {"auc_score": roc_auc_score(y_true, y_pred)}


if __name__ == "__main__":