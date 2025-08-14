import numpy as np
import polars as pl
import torch

from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from torch import nn
from typing import Union, Optional, Any

from utils.dataset_utils import download_adbench_tabular_datasets, get_data_description


def run_outlier_benchmark(
    model: Union[nn.Module, BaseEstimator],
    dataset_paths: Optional[str,Path] = None,
    predict_method: str | callable = "predict",
    random_state: int = 42,
    is_tensor: bool = True,
    device="cpu"
) -> pl.DataFrame:
    if dataset_paths is None:
        dataset_paths = Path("data/adbench_tabular_datasets")
        if not dataset_paths.exists():
            download_adbench_tabular_datasets(dataset_paths)
    else:
        dataset_paths = Path(dataset_paths)

    result_df = pl.DataFrame()

    for dataset_file in dataset_paths.glob("*.npz"):
        print(f"Running benchmark for {dataset_file.name}...")
        dataset = np.load(dataset_file)

        X = dataset['X']
        y = dataset['y']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        dataset_description = get_data_description(dataset["X"], dataset["y"], dataset_file.stem)

        if is_tensor:
            X_train = torch.from_numpy(X_train).to(device=device).float()
            X_test = torch.from_numpy(X_test).to(device=device).float()

        y_train_pred, y_test_pred = run_experiment(
            model,
            X_train,
            X_test,
            is_tensor=is_tensor,
        )

        train_scores = compute_metrics(dataset["y_train"], y_train_pred)
        test_scores = compute_metrics(dataset["y_test"], y_test_pred)


def run_experiment(
    model: Union[torch.nn.Module, BaseEstimator],
    X_train: Union[torch.Tensor, np.ndarray],
    X_test: Union[torch.Tensor, np.ndarray],
    unsup_outlier_algo: str = "local_outlier_factor",
    unsup_outlier_config: dict = None,
    is_tensor: bool = True,
):
    X_train_embed = get_embeddings(model, X_train)
    X_test_embed = get_embeddings(model, X_test)

    if is_tensor:
        X_train_embed = X_train_embed.detach().numpy()
        X_test_embed = X_test_embed.detach().numpy()

    if unsup_outlier_algo == "local_outlier_factor":
        if unsup_outlier_config is None:
            outlier_algo = LocalOutlierFactor(n_neighbors=10)
        else:
            outlier_algo = LocalOutlierFactor(**unsup_outlier_config)
    else:
        raise NotImplementedError(
            "Other unsupervised outlier detection algorithms are not implemented yet."
        )

    if type(X_train_embed) == torch.Tensor:
        if len(X_train_embed.shape) == 3:
            X_train_embed = X_train_embed.squeeze()
        X_train_embed = X_train_embed.detach().numpy()
    if type(X_test_embed) == torch.Tensor:
        if len(X_test_embed.shape) == 3:
            X_test_embed = X_test_embed.squeeze()
        X_test_embed = X_test_embed.detach().numpy()

    y_pred_train = outlier_algo.fit_predict(X_train_embed)

    y_pred_test = outlier_algo.predict(X_test_embed)

    return y_pred_train, y_pred_test


def compute_metrics(y_true, y_pred):
    raise NotImplementedError


def get_embeddings(
    model, X: Union[torch.Tensor, np.ndarray]
) -> Any | None:
    if isinstance(model, nn.Module):
        return model.forward(X)
    elif isinstance(model, BaseEstimator):
        if hasattr(model, "fit_transform"):
            return model.fit_transform(X)
        elif hasattr(model, "transform") and model.is_fitted:
            return model.transform(X)
        else:
            raise ValueError("Model is not a valid estimator.")
    else:
        raise ValueError("Model is not a valid estimator or a valid neural network model.")


def get_unsupervised_outlier_algorithm(unsup_outlier_config):
    raise NotImplementedError