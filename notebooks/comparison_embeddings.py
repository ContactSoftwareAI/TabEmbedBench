import torch
import numpy as np
import sys
import polars as pl
import pickle
from utils.tabicl_utils import get_row_embeddings_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0,"../../embedding-workflow")
import graph, load_data

model_ckpt = torch.load("../data/models/tabicl/tabicl-classifier-v1.1-0506.ckpt")

state_dict = model_ckpt["state_dict"]
config = model_ckpt["config"]

row_embedder = get_row_embeddings_model(state_dict=state_dict, config=config)
row_embedder.eval()

le = LabelEncoder()
grid_stability = pl.read_csv("../data/Data_for_UCI_named.csv")
grid_stability_features = grid_stability.select(pl.all().exclude(["stab", "stabf"]))

clf_targets = grid_stability.select("stabf")
reg_targets = grid_stability.select("stab")

train_size = int(0.9 * len(grid_stability_features))

X_train = grid_stability_features[:train_size]
X_test = grid_stability_features[train_size:]
y_train = clf_targets[:train_size]
y_test = clf_targets[train_size:]


### TabICL universal embeddings
X_train = X_train.to_torch().float().unsqueeze(0)
X_test = X_test.to_torch().float().unsqueeze(0)
y_train = le.fit_transform(y_train.to_numpy().ravel())
y_test = le.transform(y_test.to_numpy().ravel())

X_train_embed_tabicl = row_embedder(X_train)
X_test_embed_tabicl = row_embedder(X_test)


### our universal embeddings
data, categorical_columns, numerical_columns, num_bins, text_columns, num_clusters, target = load_data.grid_stability("../../data.tabdata-testsets/OpenML_CTR23_benchmark")
g = pickle.load(open(r"C:\Users\fho\Documents\code\TabData\results\Grid_stability\02_graph_building\grid_stability.graph", "rb"))
nodes = g.node_list
input_vectors = torch.tensor(np.array(g.random_embeddings(dimension=512), dtype=np.float32))
emb = {nodes[i]: np.array(input_vectors[i].cpu()) for i in range(len(nodes))}
row_embeddings = g.get_context_embedding_mean(emb)
for ind in data.index:
    row_node = "R" + g.identifier + "_" + str(ind)
    if row_node in row_embeddings:
        row_embeddings[ind] = row_embeddings.pop(row_node)
    else:
        data = data.drop(ind)
data["embeddings"] = row_embeddings
train_indices = list(data[data.set != "test"].index)
train_indices_index = {}
for i in range(len(train_indices)):
    train_indices_index[train_indices[i]] = i
test_indices = list(data[data.set == "test"].index)
data_train = data.loc[train_indices]["embeddings"]
data_test = data.loc[test_indices]["embeddings"]
X_train_embed_shellmodel = np.array(list(data_train.values))
X_test_embed_shellmodel = np.array(list(data_test.values))



score_per_neighbors_tabicl = dict()
score_per_neighbors_tabicl["neighbor_tabicl"] = []
score_per_neighbors_tabicl["roc_auc_score"] = []

score_per_neighbors_shellmodel = dict()
score_per_neighbors_shellmodel["neighbor_shellmodel"] = []
score_per_neighbors_shellmodel["roc_auc_score"] = []

for k in range(20):
    knn = KNeighborsClassifier(n_neighbors=k+1)
    knn.fit(X_train_embed_tabicl.squeeze().detach().numpy(), y_train)
    y_pred = knn.predict(X_test_embed_tabicl.squeeze().detach().numpy())
    score_per_neighbors_tabicl["neighbor_tabicl"].append(k+1)
    score_per_neighbors_tabicl["roc_auc_score"].append(roc_auc_score(y_test, y_pred))

    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(X_train_embed_shellmodel.squeeze(), y_train)
    y_pred = knn.predict(X_test_embed_shellmodel.squeeze())
    score_per_neighbors_shellmodel["neighbor_shellmodel"].append(k + 1)
    score_per_neighbors_shellmodel["roc_auc_score"].append(roc_auc_score(y_test, y_pred))

result_df_tabicl = pl.from_dict(score_per_neighbors_tabicl)
result_df_shellmodel = pl.from_dict(score_per_neighbors_shellmodel)
print(result_df_tabicl)
print(result_df_shellmodel)