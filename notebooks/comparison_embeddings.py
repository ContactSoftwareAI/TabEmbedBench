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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0,r"C:\Users\fho\Documents\code\TabData\embedding-workflow")
import graph, load_data

n_d = 2 # Embedding dimension for shell model
dataset = "grid_stability"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data and graph
if dataset == "grid_stability":
    data, categorical_columns, numerical_columns, num_bins, text_columns, num_clusters, target = load_data.grid_stability("../../data.tabdata-testsets/OpenML_CTR23_benchmark")
    g = pickle.load(open(r"C:\Users\fho\Documents\code\TabData\results\Grid_stability\02_graph_building\grid_stability.graph", "rb"))
    g_train = pickle.load(open(r"C:\Users\fho\Documents\code\TabData\results\Grid_stability\02_graph_building\grid_stability_train.graph", "rb"))
elif dataset == "titanic":
    data, categorical_columns, numerical_columns, num_bins, text_columns, num_clusters, target = load_data.load_titanic_original(
        "../../data.tabdata-testsets/Titanic")
    g = pickle.load(open(r"C:\Users\fho\Documents\code\TabData\results\Titanic\02_graph_building\titanic.graph", "rb"))
elif dataset == "synthetic":
    name = "20250331_1451_num_nodes_14"
    add = False
    target_node = ""
    max_number_of_rows = 0
    masked = 0
    noise_std = 0.1
    noise_ratio = 0.1
    hidden_dim = 2
    data, categorical_columns, numerical_columns, num_bins, text_columns, num_clusters, target = load_data.load_synthetic_data_scm(
        "../../data.tabdata-testsets/Synthetic Data SCM", name, hidden_dim, noise_std, noise_ratio, max_number_of_rows,
        add, masked)
    if len(target_node) > 0:
        target = {target_node: target[target_node]}
    print(f"Target: {target}")
    g = pickle.load(open(r"C:\Users\fho\Documents\code\TabData\results\Synthetic\02_graph_building\scm_20250331_1451_num_nodes_14_noise_std_0.1_ratio_0.1_hidden_dim_2.graph", "rb"))
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
nodes = g.node_list # same as g_train.node_list
input_vectors = (np.array(g_train.random_embeddings(dimension=n_d), dtype=np.float32))
#plt.plot(input_vectors[:, 0], input_vectors[:, 1], 'o')
#plt.show()
input_vectors = torch.tensor(input_vectors)
emb = {nodes[i]: np.array(input_vectors[i].cpu()) for i in range(len(nodes))}
row_embeddings = g.get_context_embedding_mean(emb)
for ind in data.index:
    row_node = "R" + g_train.identifier + "_" + str(ind)
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

target_key = list(target.keys())[1]
print(f"Target key: {target_key}")

data_train = data.loc[train_indices]
data_test = data.loc[test_indices]
y_train = data_train[target_key]
y_test = data_test[target_key]

# Entfernen der embeddings-Spalte für TabICL, da sie numerische Features erwartet
if dataset == "grid_stability":
    data_train_for_tabicl = data_train.drop(['stab', 'stabf', 'embeddings', 'set'], axis=1)
    data_test_for_tabicl = data_test.drop(['stab', 'stabf', 'embeddings', 'set'], axis=1)


# Konvertierung zu Tensoren für TabICL
X_train_tensor = torch.tensor(data_train_for_tabicl.values, dtype=torch.float32).unsqueeze(0).to(device)
X_test_tensor = torch.tensor(data_test_for_tabicl.values, dtype=torch.float32).unsqueeze(0).to(device)

# TabICL universal embeddings
X_train_embed_tabicl = row_embedder(X_train_tensor)
X_test_embed_tabicl = row_embedder(X_test_tensor)

# TabPFN embeddings: embeddings for every column as target
X_train_embed_tabpfn = tabpfn_model.get_embeddings(data_train_for_tabicl)
aggregated_emb_train = compute_embeddings_aggregation(X_train_embed_tabpfn, agg_func = "mean")
X_test_embed_tabpfn = tabpfn_model.get_embeddings(data_test_for_tabicl)
aggregated_emb_test = compute_embeddings_aggregation(X_test_embed_tabpfn, agg_func = "mean")

# Shell model embeddings
data_train_embeddings = data.loc[train_indices]["embeddings"]
data_test_embeddings = data.loc[test_indices]["embeddings"]
X_train_embed_shellmodel = np.array(list(data_train_embeddings.values))
X_test_embed_shellmodel = np.array(list(data_test_embeddings.values))

# Shell model embeddings test
shellmodel = ShellModel(embed_dim=n_d, g=g)
#X_combined = np.concatenate([data_train, data_test], axis=0)
X_combined = data.drop(['stab', 'stabf', 'embeddings', 'set'], axis=1)
embeddings_schalenmodell = compute_embeddings(data = X_combined, categorical_indices = [], embed_dim=n_d)
plt.plot(embeddings_schalenmodell[:, 0], embeddings_schalenmodell[:, 1], 'o')
plt.show()
X_train_embed_shellmodel2 = embeddings_schalenmodell[train_indices]#shellmodel.forward(data_train)
X_test_embed_shellmodel2 = embeddings_schalenmodell[test_indices] #shellmodel.forward(data_test)

# Label encoding
y_train = le.fit_transform(y_train.to_numpy().ravel())
y_test = le.transform(y_test.to_numpy().ravel())

# Evaluation
score_per_neighbors_tabicl = dict()
score_per_neighbors_tabicl["neighbor_tabicl"] = []
score_per_neighbors_tabicl["roc_auc_score"] = []

score_per_neighbors_tabpfn = dict()
score_per_neighbors_tabpfn["neighbor_tabpfn"] = []
score_per_neighbors_tabpfn["roc_auc_score"] = []

score_per_neighbors_shellmodel = dict()
score_per_neighbors_shellmodel["neighbor_shellmodel"] = []
score_per_neighbors_shellmodel["roc_auc_score"] = []

score_per_neighbors_shellmodel2 = dict()
score_per_neighbors_shellmodel2["neighbor_shellmodel2"] = []
score_per_neighbors_shellmodel2["roc_auc_score"] = []


# KNN
for k in [10]:
    knn1 = KNeighborsClassifier(n_neighbors=k+1, weights='distance', algorithm='ball_tree')
    knn1.fit(X_train_embed_tabicl.squeeze().detach().cpu().numpy(), y_train)
    y_pred = knn1.predict_proba(X_test_embed_tabicl.squeeze().detach().cpu().numpy())
    score_per_neighbors_tabicl["neighbor_tabicl"].append(k+1)
    score_per_neighbors_tabicl["roc_auc_score"].append(roc_auc_score(y_test, y_pred[:,1]))

    knn2 = KNeighborsClassifier(n_neighbors=k + 1, weights='distance', algorithm='ball_tree')
    knn2.fit(X_train_embed_shellmodel, y_train)
    distances, indices = knn2.kneighbors(X_test_embed_shellmodel)  # Erste 5 Test-Samples
    print("Distances to nearest neighbors (first 5 test samples):")
    for i in range(5):
        print(f"  Sample {i}: {distances[i]}")
        print(f"  Neighbor classes: {y_train[indices[i]]}")
    y_pred = knn2.predict_proba(X_test_embed_shellmodel)
    score_per_neighbors_shellmodel["neighbor_shellmodel"].append(k + 1)
    score_per_neighbors_shellmodel["roc_auc_score"].append(roc_auc_score(y_test, y_pred[:,1]))

    knn3 = KNeighborsClassifier(n_neighbors=k + 1, weights='distance', algorithm='ball_tree')
    knn3.fit(aggregated_emb_train.squeeze(), y_train)
    y_pred = knn3.predict_proba(aggregated_emb_test.squeeze())
    score_per_neighbors_tabpfn["neighbor_tabpfn"].append(k + 1)
    score_per_neighbors_tabpfn["roc_auc_score"].append(roc_auc_score(y_test, y_pred[:,1]))

    knn4 = KNeighborsClassifier(n_neighbors=k + 1, weights='distance', algorithm='ball_tree')
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
    score_per_neighbors_shellmodel2["roc_auc_score"].append(roc_auc_score(y_test, y_pred[:,1]))

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