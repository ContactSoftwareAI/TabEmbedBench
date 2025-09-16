from skrub import TableVectorizer

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator

class TabVectorizerEmbedding(BaseEmbeddingGenerator):
    def __init__(
            self,
    ):
        super().__init__()

        self.model = TableVectorizer()

    def _get_default_name(self):
        return "TabVectorizerEmbedding"

    @property
    def task_only(self):
        return False

    def preprocess_data(self, X, train=True):
        if train:
            X = self.model.fit(X)

        return X

    def compute_embeddings(self, X):
        embeddings = self.model.transform(X)
        return embeddings

    def reset_embedding_model(self):
        self.model = TableVectorizer()




