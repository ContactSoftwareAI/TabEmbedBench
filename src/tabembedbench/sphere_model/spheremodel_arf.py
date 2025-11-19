import numpy as np
from sklearn.base import TransformerMixin

from tabembedbench.utils.preprocess_utils import infer_categorical_columns


class SphereModelARF(TransformerMixin):
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
        self, embed_dim: int
    ) -> None:
        """Initialize the sphere-based embedding generator.

        Args:
            embed_dim (int): Dimensionality of the embedding space.
        """
        super()
        self.embed_dim = embed_dim
        self.categorical_indices = None
        self.column_properties = []
        self.n_cols = None

    def fit(
        self,
        data: np.ndarray,
        categorical_indices: list[int],
        y=None,
    ):
        """Fit the embedding model to the data.

        Learns embedding properties for each column, including sphere points
        for categorical features and min/max ranges for numerical features.

        Args:
            data np.ndarray: Input data to fit.
            y: Unused parameter, kept for sklearn compatibility. Defaults to None.
            categorical_indices (list[int] | None, optional): Indices of categorical
                columns. If None, will be inferred. Defaults to None.
        """
        self.categorical_indices = categorical_indices

        _, self.n_cols = data.shape

        center_points = random_points_on_unit_sphere(self.n_cols, self.embed_dim)

        for col_idx in range(self.n_cols):
            column_data = data[:, col_idx]

            if col_idx in self.categorical_indices:
                unique_categories = np.unique(column_data)
                category_embeddings = {}

                random_offsets = random_points_on_unit_sphere(len(unique_categories), self.embed_dim)

                for category in range(len(unique_categories)):
                    category_embeddings[unique_categories[category]] = center_points[col_idx] + 0.1 * random_offsets[category]

                self.column_properties.append([center_points[col_idx], category_embeddings])
            else:
                col_min = np.min(column_data)
                col_max = np.max(column_data)
                self.column_properties.append([center_points[col_idx],col_min, col_max])

    def transform(self, data: np.ndarray) -> np.ndarray:
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

    def _generate_random_sphere_point(self) -> np.ndarray:
        """Generate a random point on the unit sphere.

        Returns:
            np.ndarray: Random point on the unit sphere of dimension embed_dim.
        """
        point = np.random.randn(self.embed_dim)

        return point / np.linalg.norm(point)

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
        sphere_point = self.column_properties[col_idx][0]
        col_min = self.column_properties[col_idx][1]
        col_max = self.column_properties[col_idx][2]
        col_range = col_max - col_min

        return (0.5 + (column_data.reshape(-1,1) - col_min)/col_range) * sphere_point.reshape(1,-1)

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

        n_values = len(column_data)
        embeddings = np.zeros((n_values, self.embed_dim))

        for i, value in enumerate(column_data):
            if value in unique_category_embeddings.keys():
                embeddings[i] = unique_category_embeddings[value]
            else:
                random_offset = random_points_on_unit_sphere(1, self.embed_dim,list(unique_category_embeddings.values()))[0]

                unique_category_embeddings[value] = center_point + 0.1 * random_offset
                embeddings[i] = unique_category_embeddings[value]

        return embeddings


def create_random_unit_vector(dimension):
    norm = 1e-15
    while norm < 1e-10:
        candidate_point = np.float32(np.random.randn(dimension))
        norm = np.linalg.norm(candidate_point)
    return candidate_point/norm


def random_points_on_unit_sphere(num_points,
                                 dimension,
                                 previous_points=[]):
    num_previous_points = len(previous_points)
    points = [previous_points[i] for i in range(num_previous_points)]
    initial_separation_goal = 1
    attempts_before_reducing_separation = 500
    separation_reduction_factor = 0.5
    min_final_separation_allowed = 1e-5

    current_separation_goal = initial_separation_goal

    for i in range(num_points):
        found_point = False
        attempts_at_current_goal = 0

        while not found_point:
            candidate_point = create_random_unit_vector(dimension)
            is_separated = True

            for existing_point in points:
                distance = np.linalg.norm(candidate_point - existing_point)
                if distance < current_separation_goal:
                    is_separated = False
                    break

            if is_separated:
                points.append(candidate_point)
                found_point = True
            else:
                attempts_at_current_goal += 1
                if attempts_at_current_goal >= attempts_before_reducing_separation:
                    current_separation_goal *= separation_reduction_factor
                    current_separation_goal = max(current_separation_goal, min_final_separation_allowed)

                    attempts_at_current_goal = 0

    return points[num_previous_points:]
