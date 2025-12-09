import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from scipy.stats import qmc


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
        #self.point_generator = SobolPointGenerator(d_internal=self.embed_dim-1)
        self.categorical_column_names = None
        self.column_properties = {}
        self.n_cols = None

    def fit(
        self,
        data: pd.DataFrame,
        categorical_column_names: list[str],
        y=None,
    ):
        """Fit the embedding model to the data.

        Learns embedding properties for each column, including sphere points
        for categorical features and min/max ranges for numerical features.

        Args:
            data (pd.DataFrame): Input data to fit.
            y: Unused parameter, kept for sklearn compatibility. Defaults to None.
            categorical_column_names (list[str] | None, optional): Names of categorical
                columns. If None, will be inferred. Defaults to None.
        """
        self.categorical_column_names = categorical_column_names

        self.n_cols = len(data.columns)

        for col in data.columns:
            column_data = data[col]

            if col in self.categorical_column_names:
                unique_categories = np.unique(column_data)
                category_embeddings = {}

                for category in unique_categories:
                    category_embeddings[category] = new_point_on_unit_sphere(self.embed_dim)
                    #category_embeddings[category] = new_point(self.embed_dim)

                self.column_properties[col] = category_embeddings
            else:
                point = new_point_on_unit_sphere(self.embed_dim)
                vec = np.zeros(self.embed_dim)
                for j in range(int(np.floor(self.embed_dim / 2))):
                    vec[2*j] = point[2*j+1]
                    vec[2*j+1] = -point[2*j]
                vec /= np.linalg.norm(vec)
                col_min = np.nanmin(column_data)
                col_max = np.nanmax(column_data)
                #self.column_properties.append([point,vec,col_min, col_max])
                self.column_properties[col] = {'point': point, 'vec': vec, 'min_value': col_min, 'max_value': col_max}

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms input data into embeddings using appropriate methods for categorical and
        numerical columns and returns row-wise embeddings.

        This method processes input data to generate embeddings for each column based on
        whether the column contains categorical or numerical data. It then calculates
        row embeddings by averaging the embeddings of all columns for each row.

        Args:
            data: Input data to be transformed into embeddings. Must be either a Pandas DataFrame.

        Returns:
            np.ndarray: A NumPy array containing the row embeddings for the input data.

        Raises:
            ValueError: If the number of columns in the input data does not match the
                number of columns in the fitted data.
        """
        n_cols = len(data.columns)
        if n_cols != self.n_cols:
            raise ValueError("Number of columns in data does not match fitted data.")

        column_embeddings = []

        for col in data.columns:
            column_data = data[col]

            if col in self.categorical_column_names:
                col_embedding = self._embed_categorical_column(column_data, col)
            else:
                col_embedding = self._embed_numerical_column(column_data.to_numpy(), col)

            column_embeddings.append(col_embedding)

        row_embeddings = np.mean(np.stack(np.array(column_embeddings),axis=0),axis=0)
        #row_embeddings = np.concatenate(column_embeddings,axis=1)

        return row_embeddings

    def _embed_numerical_column(
        self, column_data: np.ndarray, col: str
    ) -> np.ndarray:
        """Embed a numerical column into the embedding space.

        Normalizes values to a radius between 0.5 and 1.5, then places points
        along a random sphere direction at the computed radius.

        Args:
            column_data (np.ndarray): The numerical data for the column.
            col (str): Name of the column being processed.

        Returns:
            np.ndarray: Embedded values of shape (len(column_data), embed_dim).
        """
        point = self.column_properties[col]['point'].reshape([1,self.embed_dim])
        vec = self.column_properties[col]['vec'].reshape([1,self.embed_dim])
        col_min = self.column_properties[col]['min_value']
        col_max = self.column_properties[col]['max_value']
        col_range = col_max - col_min

        embeddings = np.empty((len(column_data), self.embed_dim),dtype=float)
#       mit Sigmoid auf Großkreis:
#        for i, value in enumerate(column_data):
#            if value < col_min:
#                alpha = 0.2*np.pi*sig(16*(value-col_min)/col_range)
#            elif value > col_max:
#                alpha = 0.2*np.pi*(sig(16*(value-col_max)/col_range)+4)
#            else:
#                alpha = np.pi*(0.8*(value-col_min)/col_range+0.1)
#            embeddings[i, :] = np.sin(alpha) * point + np.cos(alpha) * vec
#       ohne Sigmoid auf Großkreis:
        alpha = (np.pi * (0.8 * (column_data - col_min) / col_range + 0.1)).reshape([len(column_data),1])
        embeddings = np.sin(alpha) * point + np.cos(alpha) * vec
#       auf Gerade:
#        embeddings = (0.5 + (column_data.reshape([len(column_data),1]) - col_min) / col_range) * point
        embeddings[np.argwhere(np.isnan(column_data))] = np.zeros(self.embed_dim,dtype=float)

        return np.array(embeddings,dtype=float)

    def _embed_categorical_column(
        self, column_data: pd.Series, col: str
    ) -> np.ndarray:
        """
        Embed a categorical column into the embedding space.

        Maps each category to a point on the unit sphere.
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
        unique_category_embeddings = self.column_properties[col]

        if not isinstance(unique_category_embeddings, dict):
            raise ValueError(f"The unique category embedding is not an dictionary.")

        unique_categories = np.unique(column_data)
        for value in unique_categories:
            if value not in unique_category_embeddings.keys():
                unique_category_embeddings[value] = new_point_on_unit_sphere(self.embed_dim)
                #unique_category_embeddings[value] = new_point(self.embed_dim)

        embeddings = np.empty((len(column_data), self.embed_dim),dtype=float)
        for i, value in enumerate(column_data):
            embeddings[i, :] = unique_category_embeddings[value]

        return np.array(embeddings,dtype=float)


def new_point_on_unit_sphere(d: int):
    """
    Generation of a random point on a d-dimensional unit sphere.

    Args:
        d (int): dimension of the sphere
    """
    norm = 1e-15
    while norm < 1e-10:
        candidate_point = np.float32(np.random.randn(d))
        norm = np.linalg.norm(candidate_point)
    return candidate_point/norm


def new_point(d: int):
    return np.float32(np.random.randn(d))

def random_points_on_unit_sphere(num_points: int,
                                 d: int,
                                 previous_points=[]):
    """
    Generation of num_points on a d-dimensional unit sphere which are separated from each other and from
    all given previous_points.

    Args:
        num_points (int): number of points to be generated
        d (int): dimension of the sphere
        previous points: optional list of previous points
    """
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
            candidate_point = new_point_on_unit_sphere(d)
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


class SobolPointGenerator:
    """
    Iterative generation of Sobol points in a d_internal-dimensional unit cube
    """
    def __init__(self, d_internal: int, scramble: bool = True):
        """
        Initialisation of the Sobol generator.

        Args:
            d_internal (int): dimension of the unit cube
            scramble (bool): if a random offset should be added
        """
        if d_internal < 1:
            raise ValueError("d_internal must be at least 1.")
        if d_internal > qmc.Sobol.MAXDIM:
            print(f"Warning: d_internal ({d_internal}) exceeds the maximal dimension für Sobol "
                  f"({qmc.Sobol.MAXDIM}). This can lead to bad quality.")
        self.d_internal = d_internal
        self.sampler = qmc.Sobol(d=d_internal, scramble=scramble, seed = 42)

    def get_point(self) -> np.ndarray:
        """
        Generates the next Sobol point.

        Returns:
            numpy array of coordinates
        """
        point = self.sampler.random(1)[0]
        return point


def new_point_on_unit_sphere_sobol(generator: SobolPointGenerator,
                                   d: int):
    """
    Generation of a new Sobol point on the d-dimensional unit sphere by generating a Sobol point
    in a (d-1)-dimensional unit cube and transfering these angular coordinates to cartesian ones.

    Args:
        generator (SobolPointGenerator)
        d (int): dimension of the sphere
    """
    p = generator.get_point()
    angles = [2 * np.pi * p[0]]
    for j in range(1, d-1):
        angles.append(np.acos(2 * p[j] - 1))
    cartesian_coords = np.zeros(d)
    cartesian_coords[0] = np.cos(angles[0])
    prod_sin = np.sin(angles[0])
    for i in range(1, d-1):
        cartesian_coords[i] = prod_sin * np.cos(angles[i])
        prod_sin *= np.sin(angles[i])
    cartesian_coords[d-1] = prod_sin
    return cartesian_coords


def sig(x):
    return 1/(1 + np.exp(-x))
