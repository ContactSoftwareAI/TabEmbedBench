from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
import pandas as pd


class SphereModel(TransformerMixin):
    def __init__(self, categorical_indices: List[int], embed_dim: int):
        self.categorical_indices = categorical_indices
        self.embed_dim = embed_dim
        self.column_properties = []
        self.n_cols = None
        pass

    def _generate_random_sphere_point(self) -> np.ndarray:
        """Generiert einen zufälligen Punkt auf der Einheitssphäre."""
        point = np.random.randn(self.embed_dim)
        return point / np.linalg.norm(point)

    def fit(self, data: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fits the provided data by processing each column based on its type (categorical or numerical).
        For categorical columns, a random embedding is generated for each category. For numerical
        columns, the minimum, maximum values and a random point on a sphere are stored.

        Parameters:
            data (pd.DataFrame | np.ndarray): The dataset to be fitted, provided either as a pandas
                DataFrame or a numpy array.
            y: Optional additional information, not used in this method.

        Raises:
            TypeError: Raised if the input data is neither a pandas DataFrame nor a numpy array.

        Returns:
            None
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data.copy()

        _ , self.n_cols = data_array.shape

        for col_idx in range(self.n_cols):
            column_data = data_array[:, col_idx]

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


    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transforms the input data into embeddings based on predefined logic for categorical
        and numerical data. The method can process both pandas DataFrame and NumPy ndarray
        and produces row-level embeddings by aggregating embeddings of individual columns.

        Parameters:
        data (Union[pd.DataFrame, np.ndarray]): Input data to transform. Can be either a pandas
        DataFrame or a NumPy ndarray.

        Returns:
        np.ndarray: A NumPy ndarray where each row represents the embedding of the corresponding
        row in the input data.

        Raises:
        ValueError: If the number of columns in the input data does not match the expected
        number of columns determined during fitting.
        """
        # Konvertiere zu NumPy Array falls DataFrame
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data.copy()

        n_rows, n_cols = data_array.shape
        if n_cols != self.n_cols:
            raise ValueError("Number of columns in data does not match fitted data.")
        # Erstelle Einbettungen für jede Spalte
        column_embeddings = []

        for col_idx in range(n_cols):
            column_data = data_array[:, col_idx]

            if col_idx in self.categorical_indices:
                col_embedding = self._embed_categorical_column(column_data, col_idx)
            else:
                col_embedding,  = self._embed_numerical_column(column_data, col_idx)

            column_embeddings.append(col_embedding)

        # Erstelle Zeileneinbettungen durch Mittelung
        row_embeddings = np.zeros((n_rows, self.embed_dim))
        for row_idx in range(n_rows):
            row_embedding = np.zeros(self.embed_dim)
            for col_idx in range(n_cols):
                row_embedding += column_embeddings[col_idx][row_idx]
            row_embeddings[row_idx] = row_embedding / n_cols

        return row_embeddings


    def _embed_numerical_column(self, column_data: np.ndarray, col_idx: int) -> np.ndarray:
        """
        Embeds a numerical column into a multidimensional representation based on normalization
        and a pre-calculated sphere point.

        This method processes a numerical column of data and embeds each value into a
        specified embedding dimension space. It normalizes the values within the column
        to derive the radial distance for embedding and scales the result based on
        stored column properties. The embedding is calculated by placing points
        at scaled distances along the sphere-point direction.

        Parameters:
        column_data : np.ndarray
            The input numerical data for the column to be embedded.
        col_idx : int
            The index of the column being processed, which is used to retrieve the
            corresponding column properties.

        Returns:
        np.ndarray
            A 2D numpy array of shape (len(column_data), embed_dim), representing the
            embedded values for the input column.
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
                # Normiere Wert auf Radius zwischen 0.5 und 1.5
                normalized_value = (value - col_min) / (col_max - col_min)  # 0 bis 1
                radius = 0.5 + normalized_value * 1.0  # 0.5 bis 1.5

            # Platziere Punkt auf der Linie durch den Ursprung
            embeddings[i] = radius * sphere_point

        return embeddings

    def _embed_categorical_column(self, column_data: np.ndarray, col_idx: int) -> np.ndarray:
        """
        Embeds a categorical column into a numerical array representation based on predefined
        center points and generates embeddings for unknown categories dynamically.

        Parameters
        ----------
        column_data : np.ndarray
            The categorical data of the column to be embedded.
        col_idx : int
            The index of the column in the context of all columns of the data.

        Returns
        -------
        np.ndarray
            A 2D array representing the embedded numerical values for each category
            in the input column data.
        """
        center_point = self.column_properties[col_idx][0]
        unique_category_embeddings = self.column_properties[col_idx][1]

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
