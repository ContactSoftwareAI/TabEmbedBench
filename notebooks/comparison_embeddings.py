import torch
import numpy as np
import sys
import polars as pl
import pickle
import os
import matplotlib.pyplot as plt

from tabembedbench.embedding_models.spherebasedembedding_utils import compute_embeddings
from tabembedbench.utils.shellmodel import ShellModel
from tabembedbench.embedding_models.tabicl_utils import get_row_embeddings_model
from tabpfn import TabPFNRegressor, TabPFNClassifier
from tabembedbench.embedding_models.tabpfn_utils import UniversalTabPFNEmbedding
from tabembedbench.utils.embedding_utils import compute_embeddings_aggregation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"C:\Users\fho\Documents\code\TabData\embedding-workflow")
import graph, load_data


def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error"""
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


n_d = 4  # Embedding dimension for shell model
dataset = "rossmann"  # "grid_stability"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data and graph
if dataset == "grid_stability":
    (
        data,
        categorical_columns,
        numerical_columns,
        num_bins,
        text_columns,
        num_clusters,
        target,
    ) = load_data.grid_stability("../../data.tabdata-testsets/OpenML_CTR23_benchmark")
    g = pickle.load(
        open(
            r"C:\Users\fho\Documents\code\TabData\results\Grid_stability\02_graph_building\grid_stability.graph",
            "rb",
        )
    )
    g_train = pickle.load(
        open(
            r"C:\Users\fho\Documents\code\TabData\results\Grid_stability\02_graph_building\grid_stability_train.graph",
            "rb",
        )
    )
elif dataset == "titanic":
    (
        data,
        categorical_columns,
        numerical_columns,
        num_bins,
        text_columns,
        num_clusters,
        target,
    ) = load_data.load_titanic("../../data.tabdata-testsets/Titanic")
    print(data)
    g = pickle.load(
        open(
            r"C:\Users\fho\Documents\code\TabData\results\Titanic\02_graph_building\titanic.graph",
            "rb",
        )
    )
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
    data["Age"] = data["Age"].fillna(data["Age"].mean()).round(1)
elif dataset == "rossmann":
    stores = []
    (
        data,
        categorical_columns,
        numerical_columns,
        num_bins,
        text_columns,
        num_clusters,
        target,
    ) = load_data.load_Rossmann_Store_Sales(
        "../../data.tabdata-testsets/Rossmann", stores
    )
    mean_test = np.array(data[data.set == "test"]["MeanSales"])
    g = pickle.load(
        open(
            r"C:\Users\fho\Documents\code\TabData\results\Rossmann\02_graph_building\rossmann.graph",
            "rb",
        )
    )
    data = data.drop(["Customers", "Open"], axis=1)
    data = data[data["NormalizedSales"] != 0]
    print(f"Anzahl Zeilen nach Entfernen von NormalizedSales = 0: {len(data)}")
else:
    raise ValueError("Dataset not found")
# load model for TabICL embeddings
model_ckpt = torch.load("../data/models/tabicl/tabicl-classifier-v1.1-0506.ckpt")
state_dict = model_ckpt["state_dict"]
config = model_ckpt["config"]

row_embedder = get_row_embeddings_model(state_dict=state_dict, config=config)
row_embedder = row_embedder.to(device)
row_embedder.eval()
le = LabelEncoder()

tabpfn_clf = TabPFNClassifier(device=device, n_estimators=1)
tabpfn_reg = TabPFNRegressor(device=device, n_estimators=1)

tabpfn_model = UniversalTabPFNEmbedding(tabpfn_clf, tabpfn_reg)

### our universal embeddings
nodes = g.node_list  # same as g_train.node_list
input_vectors = torch.tensor(
    np.array(g.random_embeddings(dimension=n_d), dtype=np.float32)
)
emb = {nodes[i]: np.array(input_vectors[i].cpu()) for i in range(len(nodes))}
row_embeddings = g.get_context_embedding_mean(emb)
for ind in data.index:
    row_node = "R" + g.identifier + "_" + str(ind)
    if row_node in row_embeddings:
        row_embeddings[ind] = row_embeddings.pop(row_node)
#    else:
#        data = data.drop(ind)
data["embeddings"] = row_embeddings
train_indices = list(data[data.set != "test"].index)
train_indices_index = {}
for i in range(len(train_indices)):
    train_indices_index[train_indices[i]] = i
test_indices = list(data[data.set == "test"].index)

if dataset == "grid_stability":
    target_key = list(target.keys())[1]
else:
    target_key = list(target.keys())[0]
print(f"Target key: {target_key}")

data_train = data.loc[train_indices]
data_test = data.loc[test_indices]
y_train = data_train[target_key]
y_test = data_test[target_key]

# Entfernen der embeddings-Spalte für TabICL, da sie numerische Features erwartet
if dataset == "grid_stability":
    data_train_for_tabicl = data_train.drop(
        ["stab", "stabf", "embeddings", "set"], axis=1
    )
    data_test_for_tabicl = data_test.drop(
        ["stab", "stabf", "embeddings", "set"], axis=1
    )
elif dataset == "titanic":
    data_train_for_tabicl = data_train.drop(["Survived", "embeddings", "set"], axis=1)
    data_test_for_tabicl = data_test.drop(["Survived", "embeddings", "set"], axis=1)
    # Label encoding für kategorische Spalten im Titanic Dataset
    feature_encoders = {}
    for col in categorical_columns:
        if col in data_train_for_tabicl.columns:
            encoder = LabelEncoder()
            # Fit auf den Trainingsdaten
            data_train_for_tabicl[col] = encoder.fit_transform(
                data_train_for_tabicl[col].astype(str)
            )
            # Transform auf den Testdaten
            data_test_for_tabicl[col] = encoder.transform(
                data_test_for_tabicl[col].astype(str)
            )
            feature_encoders[col] = encoder

    data_train_for_tabicl = data_train_for_tabicl.astype(np.float32)
    data_test_for_tabicl = data_test_for_tabicl.astype(np.float32)
elif dataset == "rossmann":
    data_train_for_tabicl = data_train.drop(
        ["MeanSales", "NormalizedSales", "embeddings", "set"], axis=1
    )
    data_test_for_tabicl = data_test.drop(
        ["MeanSales", "NormalizedSales", "embeddings", "set"], axis=1
    )
    feature_encoders = {}
    for col in categorical_columns:
        if col in data_train_for_tabicl.columns:
            encoder = LabelEncoder()
            # Fit auf den Trainingsdaten
            data_train_for_tabicl[col] = encoder.fit_transform(
                data_train_for_tabicl[col].astype(str)
            )
            # Transform auf den Testdaten
            data_test_for_tabicl[col] = encoder.transform(
                data_test_for_tabicl[col].astype(str)
            )
            feature_encoders[col] = encoder
else:
    raise ValueError("Dataset columns not dropped")

print(data_train_for_tabicl)

# Konvertierung zu Tensoren für TabICL
X_train_tensor = (
    torch.tensor(data_train_for_tabicl.values, dtype=torch.float32)
    .unsqueeze(0)
    .to(device)
)
X_test_tensor = (
    torch.tensor(data_test_for_tabicl.values, dtype=torch.float32)
    .unsqueeze(0)
    .to(device)
)

# TabICL universal embeddings
X_train_embed_tabicl = row_embedder(X_train_tensor)
X_test_embed_tabicl = row_embedder(X_test_tensor)

# TabPFN embeddings: embeddings for every column as target
if dataset == "grid_stability":
    X_train_embed_tabpfn = tabpfn_model.get_embeddings(data_train_for_tabicl)
    X_test_embed_tabpfn = tabpfn_model.get_embeddings(data_test_for_tabicl)
elif dataset == "titanic":
    X_train_embed_tabpfn = tabpfn_model.get_embeddings(
        data_train_for_tabicl, cat_cols=[0, 1, 6]
    )
    X_test_embed_tabpfn = tabpfn_model.get_embeddings(
        data_test_for_tabicl, cat_cols=[0, 1, 6]
    )
elif dataset == "rossmann":
    pass
    # X_train_embed_tabpfn = tabpfn_model.get_embeddings(data_train_for_tabicl, cat_cols=[0,4,5,6])
    # X_test_embed_tabpfn = tabpfn_model.get_embeddings(data_test_for_tabicl, cat_cols=[0,4,5,6])
else:
    raise ValueError("Dataset not supported")
# aggregated_emb_train = compute_embeddings_aggregation(X_train_embed_tabpfn, agg_func = "mean")
# aggregated_emb_test = compute_embeddings_aggregation(X_test_embed_tabpfn, agg_func = "mean")

# Shell model embeddings
data_train_embeddings = data.loc[train_indices]["embeddings"]
data_test_embeddings = data.loc[test_indices]["embeddings"]
X_train_embed_shellmodel = np.array(list(data_train_embeddings.values))
X_test_embed_shellmodel = np.array(list(data_test_embeddings.values))


if dataset == "grid_stability":
    X_combined = data.drop(["stab", "stabf", "embeddings", "set"], axis=1)
    embeddings_schalenmodell = compute_embeddings(
        data=X_combined, categorical_indices=[], embed_dim=n_d
    )
elif dataset == "titanic":
    X_combined = data.drop(["Survived", "embeddings", "set"], axis=1)
    embeddings_schalenmodell = compute_embeddings(
        data=X_combined, categorical_indices=[0, 1, 6], embed_dim=n_d
    )
elif dataset == "rossmann":
    X_combined = data.drop(
        ["MeanSales", "NormalizedSales", "embeddings", "set"], axis=1
    )
    embeddings_schalenmodell = compute_embeddings(
        data=X_combined, categorical_indices=[0, 4, 5, 6], embed_dim=n_d
    )
else:
    raise ValueError("Dataset not supported")

print(f"Shape von embeddings_schalenmodell: {embeddings_schalenmodell.shape}")
print(f"Anzahl train_indices: {len(train_indices)}")
print(f"Min train_index: {min(train_indices)}")
print(f"Max train_index: {max(train_indices)}")
print(f"Maximaler gültiger Index: {embeddings_schalenmodell.shape[0] - 1}")

# plt.plot(embeddings_schalenmodell[:, 0], embeddings_schalenmodell[:, 1], 'o')
# plt.show()
if dataset == "titanic":
    train_indices = [idx - 1 for idx in train_indices]
    test_indices = [idx - 1 for idx in test_indices]
elif dataset == "rossmann":
    # Erstelle Mapping von ursprünglichen Indizes zu Array-Positionen
    index_mapping = {original_idx: i for i, original_idx in enumerate(data.index)}
    train_indices = [index_mapping[idx] for idx in train_indices]
    test_indices = [index_mapping[idx] for idx in test_indices]


X_train_embed_shellmodel2 = embeddings_schalenmodell[
    train_indices
]  # shellmodel.forward(data_train)
X_test_embed_shellmodel2 = embeddings_schalenmodell[
    test_indices
]  # shellmodel.forward(data_test)

# Label encoding for classification
if dataset == "rossmann":
    pass
else:
    y_train = le.fit_transform(y_train.to_numpy().ravel())
    y_test = le.transform(y_test.to_numpy().ravel())

# Evaluation
score_per_neighbors_tabicl = dict()
score_per_neighbors_tabicl["neighbor_tabicl"] = []

score_per_neighbors_tabpfn = dict()
score_per_neighbors_tabpfn["neighbor_tabpfn"] = []

score_per_neighbors_shellmodel = dict()
score_per_neighbors_shellmodel["neighbor_shellmodel"] = []

score_per_neighbors_shellmodel2 = dict()
score_per_neighbors_shellmodel2["neighbor_shellmodel2"] = []

if dataset == "rossmann":
    score_per_neighbors_tabicl["rmspe"] = []
    score_per_neighbors_tabpfn["rmspe"] = []
    score_per_neighbors_shellmodel["rmspe"] = []
    score_per_neighbors_shellmodel2["rmspe"] = []
    for k in [10]:
        knn1 = KNeighborsRegressor(
            n_neighbors=k + 1, weights="distance", algorithm="ball_tree"
        )
        knn1.fit(X_train_embed_tabicl.squeeze().detach().cpu().numpy(), y_train)
        y_pred = knn1.predict(X_test_embed_tabicl.squeeze().detach().cpu().numpy())
        score_per_neighbors_tabicl["neighbor_tabicl"].append(k + 1)
        score_per_neighbors_tabicl["rmspe"].append(rmspe(y_test, y_pred))
        print(
            f"RMSPE for TabICL with {k} neighbors: {score_per_neighbors_tabicl['rmspe'][-1]}"
        )

        knn2 = KNeighborsRegressor(
            n_neighbors=k + 1, weights="distance", algorithm="ball_tree"
        )
        knn2.fit(X_train_embed_shellmodel, y_train)
        distances, indices = knn2.kneighbors(X_test_embed_shellmodel)
        # print("Distances to nearest neighbors (first 5 test samples):")
        # for i in [5]:
        #    print(f"  Sample {i}: {distances[i]}")
        #    print(f"  Neighbor classes: {y_train[indices[i]]}")
        y_pred = knn2.predict(X_test_embed_shellmodel)
        score_per_neighbors_shellmodel["neighbor_shellmodel"].append(k + 1)
        score_per_neighbors_shellmodel["rmspe"].append(rmspe(y_test, y_pred))
        print(
            f"RMSPE for Shell model with {k} neighbors: {score_per_neighbors_shellmodel['rmspe'][-1]}"
        )

        knn4 = KNeighborsRegressor(
            n_neighbors=k + 1, weights="distance", algorithm="ball_tree"
        )
        knn4.fit(X_train_embed_shellmodel2, y_train)
        y_pred = knn4.predict(X_test_embed_shellmodel2)
        score_per_neighbors_shellmodel2["neighbor_shellmodel2"].append(k + 1)
        score_per_neighbors_shellmodel2["rmspe"].append(rmspe(y_test, y_pred))
        print(
            f"RMSPE for Shell model with {k} neighbors: {score_per_neighbors_shellmodel2['rmspe'][-1]}"
        )
else:
    score_per_neighbors_tabicl["roc_auc_score"] = []
    score_per_neighbors_tabpfn["roc_auc_score"] = []
    score_per_neighbors_shellmodel["roc_auc_score"] = []
    score_per_neighbors_shellmodel2["roc_auc_score"] = []
    for k in [10]:
        knn1 = KNeighborsClassifier(
            n_neighbors=k + 1, weights="distance", algorithm="ball_tree"
        )
        knn1.fit(X_train_embed_tabicl.squeeze().detach().cpu().numpy(), y_train)
        y_pred = knn1.predict_proba(
            X_test_embed_tabicl.squeeze().detach().cpu().numpy()
        )
        score_per_neighbors_tabicl["neighbor_tabicl"].append(k + 1)
        score_per_neighbors_tabicl["roc_auc_score"].append(
            roc_auc_score(y_test, y_pred[:, 1])
        )

        knn2 = KNeighborsClassifier(
            n_neighbors=k + 1, weights="distance", algorithm="ball_tree"
        )
        knn2.fit(X_train_embed_shellmodel, y_train)
        distances, indices = knn2.kneighbors(
            X_test_embed_shellmodel
        )  # Erste 5 Test-Samples
        print("Distances to nearest neighbors (first 5 test samples):")
        for i in range(5):
            print(f"  Sample {i}: {distances[i]}")
            print(f"  Neighbor classes: {y_train[indices[i]]}")
        y_pred = knn2.predict_proba(X_test_embed_shellmodel)
        score_per_neighbors_shellmodel["neighbor_shellmodel"].append(k + 1)
        score_per_neighbors_shellmodel["roc_auc_score"].append(
            roc_auc_score(y_test, y_pred[:, 1])
        )

        # knn3 = KNeighborsClassifier(n_neighbors=k + 1, weights='distance', algorithm='ball_tree')
        # knn3.fit(aggregated_emb_train.squeeze(), y_train)
        # y_pred = knn3.predict_proba(aggregated_emb_test.squeeze())
        # score_per_neighbors_tabpfn["neighbor_tabpfn"].append(k + 1)
        # score_per_neighbors_tabpfn["roc_auc_score"].append(roc_auc_score(y_test, y_pred[:,1]))

        knn4 = KNeighborsClassifier(
            n_neighbors=k + 1, weights="distance", algorithm="ball_tree"
        )
        knn4.fit(X_train_embed_shellmodel2, y_train)
        # distances, indices = knn4.kneighbors(X_test_embed_shellmodel2)  # Erste 5 Test-Samples
        # #print("Distances to nearest neighbors (first 5 test samples):")
        # #for i in range(5):
        # #    print(f"  Sample {i}: {distances[i]}")
        # #    print(f"  Neighbor classes: {y_train[indices[i]]}")
        # y_pred = []
        # for i in range(len(X_test_embed_shellmodel2)):
        #     # Gewichte basierend auf inversen Distanzen
        #     weights = 1 / (distances[i] + 1e-8)  # kleine Konstante um Division durch 0 zu vermeiden
        #     # Gewichteter Durchschnitt der Nachbar-Labels
        #     neighbor_labels = np.array(y_train[indices[i]])
        #     weighted_prediction = np.sum(weights * neighbor_labels) / np.sum(weights)
        #     y_pred.append(weighted_prediction)
        #
        # y_pred = np.array(y_pred)
        # print(f"y_pred: {y_pred[:5]}")
        # y_pred = knn4.predict(X_test_embed_shellmodel2)
        # print(f"y_pred_test: {y_pred_test[:5]}")
        y_pred = knn4.predict_proba(X_test_embed_shellmodel2)
        # print(f"y_pred_proba: {y_pred_proba[:5]}")
        # #y_pred = np.matmul(1/distances, np.array(y_train[indices]))/np.sum(1/distances) #knn4.predict(X_test_embed_shellmodel2)
        score_per_neighbors_shellmodel2["neighbor_shellmodel2"].append(k + 1)
        score_per_neighbors_shellmodel2["roc_auc_score"].append(
            roc_auc_score(y_test, y_pred[:, 1])
        )

result_df_tabicl = pl.from_dict(score_per_neighbors_tabicl)
result_df_tabpfn = pl.from_dict(score_per_neighbors_tabpfn)
result_df_shellmodel = pl.from_dict(score_per_neighbors_shellmodel)
result_df_shellmodel2 = pl.from_dict(score_per_neighbors_shellmodel2)

print(result_df_tabicl)
print("Dimension 512")
print(result_df_tabpfn)
print("Dimension 192")
print(result_df_shellmodel)
print(f"Dimension {n_d}")
print(result_df_shellmodel2)
print(f"Dimension {n_d}")
