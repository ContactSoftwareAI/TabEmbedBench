import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import torch

    from utils.tabicl_utils import get_row_embeddings_model
    from sklearn.model_selection import train_test_split

    return get_row_embeddings_model, mo, pl, torch, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# TableICL Experiments""")
    return


@app.cell
def _(get_row_embeddings_model, torch):
    model_ckpt = torch.load("data/models/tabicl/tabicl-classifier-v1.1-0506.ckpt")

    state_dict = model_ckpt["state_dict"]
    config = model_ckpt["config"]

    row_embedder = get_row_embeddings_model(state_dict=state_dict, config=config)

    row_embedder.eval()

    return (row_embedder,)


@app.cell
def _(mo):
    mo.md(r"""## Grid Stability""")
    return


@app.cell
def _(pl, train_test_split):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()


    grid_stability = pl.read_csv("./data/Data_for_UCI_named.csv")

    grid_stability_features = grid_stability.select(pl.all().exclude(["stab", "stabf"]))

    clf_targets = grid_stability.select("stabf")
    reg_targets = grid_stability.select("stab")

    X_train, X_test, y_train, y_test = train_test_split(
        grid_stability_features,
        clf_targets,
        test_size=0.2,
        random_state=42
    )

    X_train = X_train.to_torch().float().unsqueeze(0)

    X_test = X_test.to_torch().float().unsqueeze(0)

    y_train = le.fit_transform(y_train.to_numpy().ravel())

    y_test = le.transform(y_test.to_numpy().ravel())
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, row_embedder):
    X_train_embed = row_embedder(X_train)
    X_test_embed = row_embedder(X_test)
    return X_test_embed, X_train_embed


@app.cell
def _(X_test_embed, X_train_embed, pl, y_test, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score

    score_per_neighbors = dict()

    score_per_neighbors["neighbor"] = []
    score_per_neighbors["roc_auc_score"] = []

    for k in range(150):
        knn = KNeighborsClassifier(n_neighbors=k+1)
    
        knn.fit(X_train_embed.squeeze().detach().numpy(), y_train)
    
        y_pred = knn.predict(X_test_embed.squeeze().detach().numpy())
    
        score_per_neighbors["neighbor"].append(k+1)
        score_per_neighbors["roc_auc_score"].append(roc_auc_score(y_test, y_pred))

    result_df = pl.from_dict(score_per_neighbors)
    result_df
    return (result_df,)


@app.cell
def _(result_df):
    import plotly.express as px

    px.line(data_frame=result_df, x="neighbor", y="roc_auc_score")
    return


@app.cell
def _(mo):
    mo.md(r"""## Outlier Detection""")
    return


@app.cell
def _():
    import numpy as np

    thyroid_data = np.load("data/adbench_tabular_datasets/38_thyroid.npz")

    thyroid_data
    return


@app.cell
def _():
    from sklearn.neighbors import LocalOutlierFactor
    return


if __name__ == "__main__":
    app.run()
