import torch
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.insert(0,"../../embedding-workflow")
import graph, load_data



class ShellModel():
    def __init__(self, embed_dim: int):
        self.embed_dim = 512


    def forward(self, data: pd.DataFrame, g_train: graph, g: graph) -> np.array:
        nodes = g.node_list
        input_vectors = torch.tensor(np.array(g_train.random_embeddings(dimension=self.embed_dim), dtype=np.float32))
        emb = {nodes[i]: np.array(input_vectors[i].cpu()) for i in range(len(nodes))}
        row_embeddings = g.get_context_embedding_mean(emb)
        filtered_row_embeddings = {} # Neue gefilterte row_embeddings nur für die aktuellen Datenzeilen
        for ind in data.index:
            row_node = "R" + g.identifier + "_" + str(ind)
            if row_node in row_embeddings:
                filtered_row_embeddings[ind] = row_embeddings[row_node]
                #row_embeddings[ind] = row_embeddings.pop(row_node)
            else:
                data = data.drop(ind)
        return np.array(list(filtered_row_embeddings.values()))