import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, r"C:\Users\fho\Documents\code\TabData\embedding-workflow")
#sys.path.insert(0,"../../embedding-workflow")
import graph, load_data



class ShellModel():
    def __init__(self, embed_dim: int, g: graph):
        self.embed_dim = embed_dim
        self.g = g
        self.nodes = self.g.node_list
        self.input_vectors = np.array(self.g.random_embeddings(dimension=self.embed_dim), dtype=np.float32)
        #plt.plot(self.input_vectors[:, 0], self.input_vectors[:, 1], 'o')
        #plt.show()
        emb = {self.nodes[i]: np.array(self.input_vectors[i]) for i in range(len(self.nodes))}
        self.row_embeddings = g.get_context_embedding_mean(emb)


    def forward(self, data: pd.DataFrame) -> np.array:

        filtered_row_embeddings = {} # Neue gefilterte row_embeddings nur für die aktuellen Datenzeilen
        for ind in data.index:
            row_node = "R" + self.g.identifier + "_" + str(ind)
            if row_node in self.row_embeddings:
                filtered_row_embeddings[ind] = self.row_embeddings[row_node]
                #row_embeddings[ind] = row_embeddings.pop(row_node)
            else:
                data = data.drop(ind)
        liste = list(filtered_row_embeddings.values())
        return np.array(liste)