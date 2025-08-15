import torch
import numpy as np
import sys
import polars as pl
import pickle
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.insert(0, src_dir)
from utils.shellmodel import ShellModel
from utils.tabicl_utils import get_row_embeddings_model
from tabpfn import TabPFNRegressor, TabPFNClassifier
from utils.tabpfn_utils import UniversalTabPFNEmbedding
from utils.embedding_utils import get_embeddings_aggregation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0,"../../embedding-workflow")
import graph, load_data

n_d = 512 # Embedding dimension for shell model
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
input_vectors = torch.tensor(np.array(g_train.random_embeddings(dimension=n_d), dtype=np.float32))
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

data_train = data.loc[train_indices]
data_test = data.loc[test_indices]
y_train = data_train[target_key]
y_test = data_test[target_key]

# Entfernen der embeddings-Spalte für TabICL, da sie numerische Features erwartet
if dataset == "grid_stability":
    data_train_for_tabicl = data_train.drop(['stab', 'stabf', 'embeddings', 'set'], axis=1)
    data_test_for_tabicl = data_test.drop(['stab', 'stabf', 'embeddings', 'set'], axis=1)
#elif dataset == "synthetic":
#    print(data_train)
#    data_train_for_tabicl = data_train.drop(['stab', 'stabf', 'embeddings', 'set'], axis=1)
#    data_test_for_tabicl = data_test.drop(['stab', 'stabf', 'embeddings', 'set'], axis=1)

# Konvertierung zu Tensoren für TabICL
X_train_tensor = torch.tensor(data_train_for_tabicl.values, dtype=torch.float32).unsqueeze(0).to(device)
X_test_tensor = torch.tensor(data_test_for_tabicl.values, dtype=torch.float32).unsqueeze(0).to(device)

# TabICL universal embeddings
X_train_embed_tabicl = row_embedder(X_train_tensor)
X_test_embed_tabicl = row_embedder(X_test_tensor)

# TabPFN embeddings: embeddings for every column as target
X_train_embed_tabpfn = tabpfn_model.get_embeddings(data_train_for_tabicl)
aggregated_emb_train = get_embeddings_aggregation(X_train_embed_tabpfn, agg_func = "mean")
X_test_embed_tabpfn = tabpfn_model.get_embeddings(data_test_for_tabicl)
aggregated_emb_test = get_embeddings_aggregation(X_test_embed_tabpfn, agg_func = "mean")

# Shell model embeddings
data_train_embeddings = data.loc[train_indices]["embeddings"]
data_test_embeddings = data.loc[test_indices]["embeddings"]
X_train_embed_shellmodel = np.array(list(data_train_embeddings.values))
X_test_embed_shellmodel = np.array(list(data_test_embeddings.values))

# Shell model embeddings test
shellmodel = ShellModel(embed_dim=n_d)
X_train_embed_shellmodel2 = shellmodel.forward(data_train, g_train, g)
X_test_embed_shellmodel2 = shellmodel.forward(data_test, g_train, g)

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
for k in range(10):
    knn = KNeighborsClassifier(n_neighbors=k+1)
    knn.fit(X_train_embed_tabicl.squeeze().detach().cpu().numpy(), y_train)
    y_pred = knn.predict(X_test_embed_tabicl.squeeze().detach().cpu().numpy())
    score_per_neighbors_tabicl["neighbor_tabicl"].append(k+1)
    score_per_neighbors_tabicl["roc_auc_score"].append(roc_auc_score(y_test, y_pred))

    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(X_train_embed_shellmodel, y_train)
    y_pred = knn.predict(X_test_embed_shellmodel)
    score_per_neighbors_shellmodel["neighbor_shellmodel"].append(k + 1)
    score_per_neighbors_shellmodel["roc_auc_score"].append(roc_auc_score(y_test, y_pred))

    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(aggregated_emb_train.squeeze(), y_train)
    y_pred = knn.predict(aggregated_emb_test.squeeze())
    score_per_neighbors_tabpfn["neighbor_tabpfn"].append(k + 1)
    score_per_neighbors_tabpfn["roc_auc_score"].append(roc_auc_score(y_test, y_pred))

    knn = KNeighborsClassifier(n_neighbors=k + 1)
    knn.fit(X_train_embed_shellmodel2, y_train)
    y_pred = knn.predict(X_test_embed_shellmodel2)
    score_per_neighbors_shellmodel2["neighbor_shellmodel2"].append(k + 1)
    score_per_neighbors_shellmodel2["roc_auc_score"].append(roc_auc_score(y_test, y_pred))

result_df_tabicl = pl.from_dict(score_per_neighbors_tabicl)
result_df_tabpfn = pl.from_dict(score_per_neighbors_tabpfn)
result_df_shellmodel = pl.from_dict(score_per_neighbors_shellmodel)
result_df_shellmodel2 = pl.from_dict(score_per_neighbors_shellmodel2)

print(result_df_tabicl)
print("Dimension 512")
print(result_df_tabpfn)
print("Dimension 192")
print(result_df_shellmodel)
print("Dimension 512")
print(result_df_shellmodel2)
print("Dimension 512")


# Histogramme für die drei Embedding-Methoden
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Histogramme der Embedding-Komponenten für alle drei Methoden', fontsize=16)

# 1. TabICL Embeddings
tabicl_train_flat = X_train_embed_tabicl.squeeze().detach().cpu().numpy().flatten()
tabicl_test_flat = X_test_embed_tabicl.squeeze().detach().cpu().numpy().flatten()
tabicl_all = np.concatenate([tabicl_train_flat, tabicl_test_flat])

axes[0].hist(tabicl_all, bins=150, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_title('TabICL Embeddings')
axes[0].set_xlabel('Embedding-Werte')
axes[0].set_ylabel('Häufigkeit')
axes[0].grid(True, alpha=0.3)

# 2. TabPFN Embeddings
tabpfn_train_flat = aggregated_emb_train.squeeze().flatten()
tabpfn_test_flat = aggregated_emb_test.squeeze().flatten()
tabpfn_all = np.concatenate([tabpfn_train_flat, tabpfn_test_flat])

axes[1].hist(tabpfn_all, bins=150, alpha=0.7, color='red', edgecolor='black')
axes[1].set_title('TabPFN Embeddings')
axes[1].set_xlabel('Embedding-Werte')
axes[1].set_ylabel('Häufigkeit')
axes[1].grid(True, alpha=0.3)

# 3. Shell Model Embeddings
shellmodel_train_flat = X_train_embed_shellmodel.squeeze().flatten()
shellmodel_test_flat = X_test_embed_shellmodel.squeeze().flatten()
shellmodel_all = np.concatenate([shellmodel_train_flat, shellmodel_test_flat])

axes[2].hist(shellmodel_all, bins=150, alpha=0.7, color='green', edgecolor='black')
axes[2].set_title('Shell Model Embeddings')
axes[2].set_xlabel('Embedding-Werte')
axes[2].set_ylabel('Häufigkeit')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Zusätzlich: Statistiken für jede Methode ausgeben
print("\n=== Embedding Statistiken ===")
print(f"TabICL - Min: {tabicl_all.min():.4f}, Max: {tabicl_all.max():.4f}, Mean: {tabicl_all.mean():.4f}, Std: {tabicl_all.std():.4f}")
print(f"TabPFN - Min: {tabpfn_all.min():.4f}, Max: {tabpfn_all.max():.4f}, Mean: {tabpfn_all.mean():.4f}, Std: {tabpfn_all.std():.4f}")
print(f"Shell Model - Min: {shellmodel_all.min():.4f}, Max: {shellmodel_all.max():.4f}, Mean: {shellmodel_all.mean():.4f}, Std: {shellmodel_all.std():.4f}")
