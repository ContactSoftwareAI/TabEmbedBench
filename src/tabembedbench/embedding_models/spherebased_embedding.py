import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator
from tabembedbench.utils.preprocess_utils import infer_categorical_columns


class SphereBasedEmbedding(TransformerMixin, AbstractEmbeddingGenerator):
    """Sphere-based embedding generator for tabular data.

    This embedding model projects tabular data onto high-dimensional spheres,
    treating categorical and numerical features differently. Categorical features
    are embedded as points in small regions around random sphere points, while
    numerical features are embedded along radial directions.

    Attributes:
        embed_dim (int): Dimensionality of the embedding space.
        categorical_indices (list[int] | None): Indices of categorical columns.
        column_properties (list): List storing embedding properties for each column.
        n_cols (int | None): Number of columns in the fitted data.
    """

    def __init__(
        self, embed_dim: int, categorical_indices: list[int] | None = None
    ) -> None:
        """Initialize the sphere-based embedding generator.

        Args:
            embed_dim (int): Dimensionality of the embedding space.
            categorical_indices (list[int] | None, optional): Indices of categorical
                columns. If None, will be inferred automatically. Defaults to None.
        """
        super().__init__(name=f"Sphere-Based (Dim {embed_dim})")
        self.categorical_indices = categorical_indices
        self.embed_dim = embed_dim
        self.column_properties = []
        self.n_cols = None

    @property
    def task_only(self) -> bool:
        """Indicates whether this model requires task labels.

        Returns:
            bool: False, as this is an unsupervised embedding method.
        """
        return False

    def _generate_random_sphere_point(self) -> np.ndarray:
        """Generate a random point on the unit sphere.

        Returns:
            np.ndarray: Random point on the unit sphere of dimension embed_dim.
        """
        point = np.random.randn(self.embed_dim)

        return point / np.linalg.norm(point)

    def fit(
        self,
        data: pd.DataFrame | np.ndarray,
        y=None,
        categorical_indices: list[int] | None = None,
    ):
        """Fit the embedding model to the data.

        Learns embedding properties for each column, including sphere points
        for categorical features and min/max ranges for numerical features.

        Args:
            data (pd.DataFrame | np.ndarray): Input data to fit.
            y: Unused parameter, kept for sklearn compatibility. Defaults to None.
            categorical_indices (list[int] | None, optional): Indices of categorical
                columns. If None, will be inferred. Defaults to None.

        Returns:
            self: The fitted embedding model.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        else:
            data = data

        if categorical_indices is None:
            self.categorical_indices = infer_categorical_columns(data)
        else:
            self.categorical_indices = categorical_indices

        _, self.n_cols = data.shape

        for col_idx in range(self.n_cols):
            column_data = data[:, col_idx]

            if col_idx in self.categorical_indices:
                unique_categories = np.unique(column_data)
                center_point = self._generate_random_sphere_point()
                category_embeddings = {}

                for category in unique_categories:
                    # Generiere zufälligen Punkt in kleiner Kugel (Radius 0.1) um Mittelpunkt
                    random_offset = np.random.randn(self.embed_dim)
                    random_offset = 0.1 * random_offset / np.linalg.norm(random_offset)

                    category_embeddings[category] = center_point + random_offset

                self.column_properties.append([center_point, category_embeddings])
            else:
                col_min = np.min(column_data)
                col_max = np.max(column_data)
                sphere_point = self._generate_random_sphere_point()
                self.column_properties.append([col_min, col_max, sphere_point])

    def transform(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms input data into embeddings using appropriate methods for categorical and
        numerical columns and returns row-wise embeddings.

        This method processes input data to generate embeddings for each column based on
        whether the column contains categorical or numerical data. It then calculates
        row embeddings by averaging the embeddings of all columns for each row.

        Args:
            data: Input data to be transformed into embeddings. Must be either a Pandas
                DataFrame or a NumPy ndarray.

        Returns:
            np.ndarray: A NumPy array containing the row embeddings for the input data.

        Raises:
            ValueError: If the number of columns in the input data does not match the
                number of columns in the fitted data.
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        n_rows, n_cols = data_array.shape
        if n_cols != self.n_cols:
            raise ValueError("Number of columns in data does not match fitted data.")

        column_embeddings = []

        for col_idx in range(n_cols):
            column_data = data_array[:, col_idx]

            if col_idx in self.categorical_indices:
                col_embedding = self._embed_categorical_column(column_data, col_idx)
            else:
                col_embedding = self._embed_numerical_column(column_data, col_idx)

            column_embeddings.append(col_embedding)

        # Erstelle Zeileneinbettungen durch Mittelung
        row_embeddings = np.zeros((n_rows, self.embed_dim))
        for row_idx in range(n_rows):
            row_embedding = np.zeros(self.embed_dim)
            for col_idx in range(n_cols):
                row_embedding += column_embeddings[col_idx][row_idx]
            row_embeddings[row_idx] = row_embedding / n_cols

        return row_embeddings

    def _embed_numerical_column(
        self, column_data: np.ndarray, col_idx: int
    ) -> np.ndarray:
        """Embed a numerical column into the embedding space.

        Normalizes values to a radius between 0.5 and 1.5, then places points
        along a random sphere direction at the computed radius.

        Args:
            column_data (np.ndarray): The numerical data for the column.
            col_idx (int): Index of the column being processed.

        Returns:
            np.ndarray: Embedded values of shape (len(column_data), embed_dim).
        """
        embeddings = np.zeros((len(column_data), self.embed_dim))
        col_min = self.column_properties[col_idx][0]
        col_max = self.column_properties[col_idx][1]
        sphere_point = self.column_properties[col_idx][2]

        for i, value in enumerate(column_data):
            if col_max == col_min:
                # Alle Werte sind gleich - verwende Mittelpunkt
                radius = 1.0
            else:
                col_max_64 = np.float64(col_max)
                value_64 = np.float64(value)
                col_min_64 = np.float64(col_min)
                range_val_64 = col_max_64 - col_min_64

                # Normiere Wert auf Radius zwischen 0.5 und 1.5
                normalized_value = (value_64 - col_min_64) / range_val_64
                # 0 bis 1
                radius = 0.5 + normalized_value * 1.0  # 0.5 bis 1.5

            # Platziere Punkt auf der Linie durch den Ursprung
            embeddings[i] = radius * sphere_point

        return embeddings

    def _embed_categorical_column(
        self, column_data: np.ndarray, col_idx: int
    ) -> np.ndarray:
        """Embed a categorical column into the embedding space.

        Maps each category to a point in a small region around a random center point.
        Unknown categories encountered during transform are assigned new points
        dynamically.

        Args:
            column_data (np.ndarray): The categorical data for the column.
            col_idx (int): Index of the column being processed.

        Returns:
            np.ndarray: Embedded values of shape (len(column_data), embed_dim).

        Raises:
            ValueError: If unique_category_embeddings is not a dictionary.
        """
        center_point = self.column_properties[col_idx][0]
        unique_category_embeddings = self.column_properties[col_idx][1]

        if not isinstance(unique_category_embeddings, dict):
            raise ValueError(f"The unique category embedding is not an dictionary.")

        # Erstelle Einbettungen für alle Werte
        n_values = len(column_data)
        embeddings = np.zeros((n_values, self.embed_dim))

        for i, value in enumerate(column_data):
            if value in unique_category_embeddings.keys():
                embeddings[i] = unique_category_embeddings[value]
            else:
                random_offset = np.random.randn(self.embed_dim)
                random_offset = 0.1 * random_offset / np.linalg.norm(random_offset)

                unique_category_embeddings[value] = center_point + random_offset
                embeddings[i] = unique_category_embeddings[value]

        return embeddings

    def _preprocess_data(
        self,
        data: np.ndarray,
        train: bool = True,
        outlier: bool = False,
        categorical_indices: list[int] | None = None,
    ):
        """Preprocess input data (no-op for this model).

        Args:
            data (np.ndarray): Input data.
            train (bool, optional): Whether this is training data. Defaults to True.
            outlier (bool, optional): Whether this is outlier data. Defaults to False.
            categorical_indices (list[int] | None, optional): Categorical column indices.
                Defaults to None.

        Returns:
            np.ndarray: The input data unchanged.
        """
        return data

    def _fit_model(
        self,
        data: np.ndarray,
        train: bool = True,
        categorical_indices: list[int] | None = None,
        **kwargs,
    ):
        """Fit the embedding model to preprocessed data.

        Args:
            data (np.ndarray): Preprocessed input data.
            train (bool, optional): Whether to fit the model. Defaults to True.
            categorical_indices (list[int] | None, optional): Categorical column indices.
                Defaults to None.
            **kwargs: Additional keyword arguments (unused).
        """
        if train:
            self.fit(data, categorical_indices=categorical_indices)

    def _compute_embeddings(self, data: np.ndarray, **kwargs):
        """Compute embeddings for the input data.

        Args:
            data (np.ndarray): Input data to embed.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Embeddings of shape (n_samples, embed_dim).
        """
        return self.transform(data)

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Clears all fitted column properties and metadata.
        """
        self.column_properties = []
        self.categorical_indices = None
        self.n_cols = None
