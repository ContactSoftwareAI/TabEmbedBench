import torch

from sklearn.neighbors import LocalOutlierFactor


def run_benchmark(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
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

    outlier_algo.fit(X_train_embed.squeeze().detach().numpy())

    y_pred = outlier_algo.predict(X_test_embed.squeeze().detach().numpy())

    return y_pred, outlier_algo.negative_outlier_factor_


def get_embeddings(model, X: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward"):
        return model.forward(X)
    if hasattr(model, "transform"):
        return model.transform(X)
