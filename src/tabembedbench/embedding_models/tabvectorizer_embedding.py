from skrub import TableVectorizer
import polars as pl

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator

class TabVectorizerEmbedding(BaseEmbeddingGenerator):
    """Handles the embedding generation using a TableVectorizer model.

    This class provides methods for preprocessing data, generating embeddings,
    and resetting the embedding model. It is designed to work with tabular data
    and leverages the TableVectorizer for embedding generation.

    Attributes:
        model (TableVectorizer): Instance of the TableVectorizer model used for
            embedding generation.
    """
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

    def _preprocess_data(self, X, train=True):
        """
        Preprocesses the input data by converting it from a NumPy array to a
        specific format and optionally fitting it using the model.

        Args:
            X: numpy.ndarray
                The input data to be processed.
            train: bool, optional
                A flag indicating if the model should be fitted with the input
                data (default is True).

        Returns:
            Any:
                The processed input data in the required format.
        """
        X = pl.from_numpy(X)

        if train:
            self.model.fit(X)

        return X

    def _compute_embeddings(self, X):
        embeddings = self.model.transform(X)

        return embeddings.to_numpy()

    def reset_embedding_model(self):
        self.model = TableVectorizer()




