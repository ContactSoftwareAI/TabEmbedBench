import numpy as np
import polars as pl
import torch

from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator
from typing import Union


def run_benchmark(model: Union[torch.nn.Module, BaseEstimator]) -> pl.DataFrame:
    pass


def run_experiment(
    model: Union[torch.nn.Module, BaseEstimator],
    X_train: Union[torch.Tensor, np.ndarray],
    y_train: Union[torch.Tensor, np.ndarray],
    X_test: Union[torch.Tensor, np.ndarray],
    y_test: Union[torch.Tensor, np.ndarray],
    dataset_description: dict,
    unsup_outlier_algo: str = "local_outlier_factor",
    unsup_outlier_config: dict = None,
):
    X_train_embed = get_embeddings(model, X_train)
    X_test_embed = get_embeddings(model, X_test)

    if unsup_outlier_algo == "local_outlier_factor":
        if unsup_outlier_config is None:
            outlier_algo = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
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

    outlier_algo.fit(X_train_embed)

    y_pred = outlier_algo.predict(X_test_embed)

    return y_pred, outlier_algo.negative_outlier_factor_


def get_embeddings(
    model, X: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    if hasattr(model, "forward"):
        return model.forward(X)
    elif hasattr(model, "transform"):
        if model.is_fitted:
            return model.transform(X)
        else:
            return model.fit_transform(X)
    else:
        raise ValueError("Model does not have a forward or transform method.")


def get_unsupervised_outlier_algorithm(unsup_outlier_config):
    raise NotImplementedError
