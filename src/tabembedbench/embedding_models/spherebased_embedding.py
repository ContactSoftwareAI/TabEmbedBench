import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import TransformerMixin

from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator


class SphereBasedEmbedding(AbstractEmbeddingGenerator):
    """Sphere-based embedding generator for tabular data.

    This embedding model projects tabular data onto high-dimensional spheres,
    treating categorical and numerical features differently. Categorical features
    are embedded as random points on the unit sphere, while numerical features are
    embedded along a great circle.

    Attributes:
        embed_dim (int): Dimensionality of the embedding space.
        categorical_indices (list[int] | None): Indices of categorical columns.
        column_properties (list): List storing embedding properties for each column.
        n_cols (int | None): Number of columns in the fitted data.
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize the sphere-based embedding generator.

        Args:
            embed_dim (int): Dimensionality of the embedding space.
        """
        super().__init__(name=f"Sphere-Based (Dim {embed_dim})")
        self.categorical_indices = None
        self.embed_dim = embed_dim
        self.column_properties = []
        self.n_cols = None
        self.model = SphereModel(embed_dim=embed_dim)

    def _preprocess_data(
        self,
        X: np.ndarray | pl.DataFrame | pd.DataFrame,
        train: bool = True,
        outlier: bool = False,
        categorical_column_names: list[str] = [],
        categorical_indices: list[int] = [],
        **kwargs,
    ) -> pd.DataFrame:
        """Preprocess input data. Will transform the data to the common type pd.DataFrame for further processing.

        Args:
            X (np.ndarray | pl.DataFrame | pd.DataFrame): Input data to preprocess.
            train (bool, optional): Whether this is training mode. Defaults to True.
            outlier (bool, optional): Whether to handle outliers. Defaults to False.
            categorical_column_names: Names of the categorical columns,
            categorical_indices: Indices of the categorical columns,
            **kwargs: Additional keyword arguments (unused).

        Returns:
            pd.DataFrame
        """
        if isinstance(X, np.ndarray):
            temp_data_list: list[pd.Series] = []
            column_names: list[str] = []

            for i in range(X.shape[1]):
                col_series = pd.Series(X[:, i])

                if i in categorical_indices:
                    column_names.append(f"categorical_{i}")
                    # Keep as-is, Pandas will likely assign object dtype for mixed/string
                    temp_data_list.append(col_series)
                else:
                    column_names.append(f"numerical_{i}")
                    # Ensure it's float for NaN handling
                    numeric_col_series = pd.to_numeric(
                        col_series, errors="coerce"
                    ).astype(float)

                    # Handle inf/-inf by replacing them with NaN
                    numeric_col_series.replace([np.inf, -np.inf], np.nan, inplace=True)

                    temp_data_list.append(numeric_col_series)

            processed_df = pd.concat(temp_data_list, axis=1)
            processed_df.columns = column_names

        elif isinstance(X, pd.DataFrame):
            processed_cols = {}

            for col_index, (col_name, dtype) in enumerate(X.dtypes.items()):
                col_series = X[col_name]

                if col_name in categorical_column_names:
                    processed_cols[col_name] = col_series
                else:
                    if pd.api.types.is_numeric_dtype(dtype):
                        processed_cols[col_name] = col_series.replace(
                            [np.inf, -np.inf], np.nan
                        )
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        processed_cols[col_name] = (
                            col_series.astype(np.int64) // 10**9
                        ).astype(float)
                    else:
                        try:
                            processed_cols[col_name] = col_series.astype(float)
                        except:
                            processed_cols[col_name] = col_series
                            categorical_column_names.append(str(col_name))
                            categorical_indices.append(int(col_index))

            processed_df = pd.DataFrame(processed_cols, index=X.index)

        elif isinstance(X, pl.DataFrame):
            processed_df = X.to_pandas()

            processed_pd_dfs = []

            for col_index, (col_name, dtype) in enumerate(processed_df.dtypes.items()):
                col_series = processed_df[col_name]
                if col_name in categorical_column_names:
                    processed_pd_dfs.append(col_series.to_frame())
                else:
                    if pd.api.types.is_numeric_dtype(dtype):
                        col_series.replace([np.inf, -np.inf], np.nan, inplace=True)
                        processed_pd_dfs.append(col_series.to_frame())
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        processed_pd_dfs.append(
                            (col_series.astype(np.int64) // 10**9)
                            .astype(float)
                            .to_frame()
                        )
                    else:
                        try:
                            processed_pd_dfs.append(col_series.astype(float).to_frame())
                        except:
                            processed_pd_dfs.append(col_series.to_frame())
                            categorical_column_names.append(col_name)
                            categorical_indices.append(int(col_index))

            if processed_pd_dfs:
                processed_df = pd.concat(processed_pd_dfs, axis=1)
            else:
                processed_df = pd.DataFrame()

        else:
            raise TypeError(
                f"Unsupported data type: {type(X)}. Expected np.ndarray, pl.DataFrame or pd.DataFrame."
            )
        return processed_df

    def _fit_model(
        self,
        data: pd.DataFrame,
        categorical_column_names: list[str] | None = None,
        **kwargs,
    ):
        """Fit the embedding model to preprocessed data.

        Args:
            data (pd:DataFrame): Preprocessed input data.
            categorical_column_names (list[str] | None, optional): Names of categorical columns.
                Defaults to None.
            **kwargs: Additional keyword arguments (unused).
        """
        if categorical_column_names is None:
            categorical_column_names = []
        self.model.fit(data, categorical_column_names=categorical_column_names)

    def _compute_embeddings(
        self,
        X_train_preprocessed: pd.DataFrame,
        X_test_preprocessed: pd.DataFrame | None = None,
        outlier: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute embeddings for the input data.

        Args:
            data (pd.DataFrame): Input data to embed.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            np.ndarray: Embeddings of shape (n_samples, embed_dim).
        """
        embeddings_train = self.model.transform(X_train_preprocessed)
        embeddings_test = (
            None
            if X_test_preprocessed is None
            else self.model.transform(X_test_preprocessed)
        )

        return embeddings_train, embeddings_test

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Clears all fitted column properties and metadata.
        """
        self.model = SphereModel(self.embed_dim)


class SphereModel(TransformerMixin):
    """Sphere-based embedding generator for tabular data.

    This embedding model projects tabular data onto a high-dimensional unit sphere,
    treating categorical and numerical features differently. Categorical features
    are embedded as random points on the unit sphere, while numerical features are
    embedded along a great circle.

    Attributes:
        embed_dim (int): Dimensionality of the embedding space.
        categorical_column_names (list[str]): Names of categorical columns.
        column_properties (list): List storing embedding properties for each column.
        n_cols (int | None): Number of columns in the fitted data.
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize the sphere-based embedding generator.

        Args:
            embed_dim (int): Dimensionality of the embedding space.
        """
        super()
        self.embed_dim = embed_dim
        # self.point_generator = SobolPointGenerator(d_internal=self.embed_dim-1)
        self.categorical_column_names = None
        self.column_properties = {}
        self.n_cols = None

    def fit(
        self,
        data: pd.DataFrame,
        categorical_column_names: list[str] = [],
        y=None,
    ):
        """Fit the embedding model to the data.

        Learns embedding properties for each column, including sphere points
        for categorical features and min/max ranges for numerical features.

        Args:
            data (pd.DataFrame): Input data to fit.
            categorical_column_names (list[str], optional): Names of categorical columns.
            y: Unused parameter, kept for sklearn compatibility. Defaults to None.
        """
        self.categorical_column_names = categorical_column_names

        self.n_cols = len(data.columns)

        for col in data.columns:
            column_data = data[col]

            if col in self.categorical_column_names:
                unique_categories = column_data.dropna().unique()
                category_embeddings = {}

                for category in unique_categories:
                    category_embeddings[category] = new_point_on_unit_sphere(
                        self.embed_dim
                    )
                    # category_embeddings[category] = new_point(self.embed_dim)

                self.column_properties[col] = category_embeddings
            else:
                point = new_point_on_unit_sphere(self.embed_dim)
                vec = np.zeros(self.embed_dim)
                for j in range(int(np.floor(self.embed_dim / 2))):
                    vec[2 * j] = point[2 * j + 1]
                    vec[2 * j + 1] = -point[2 * j]
                vec /= np.linalg.norm(vec)
                col_min = float(np.nanmin(column_data))
                col_max = float(np.nanmax(column_data))
                # self.column_properties.append([point,vec,col_min, col_max])
                self.column_properties[col] = {
                    "point": point,
                    "vec": vec,
                    "min_value": col_min,
                    "max_value": col_max,
                }

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transforms input data into embeddings using appropriate methods for categorical and
        numerical columns and returns row-wise embeddings.

        This method processes input data to generate embeddings for each column based on
        whether the column contains categorical or numerical data. It then calculates
        row embeddings by averaging the embeddings of all columns for each row.

        Args:
            data: Input data to be transformed into embeddings. Must be a Pandas DataFrame.

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
                col_embedding = self._embed_numerical_column(
                    column_data.to_numpy(), col
                )

            column_embeddings.append(col_embedding)

        row_embeddings = np.mean(np.stack(np.array(column_embeddings), axis=0), axis=0)
        # row_embeddings = np.concatenate(column_embeddings,axis=1)

        return row_embeddings

    def _embed_numerical_column(self, column_data: np.ndarray, col: str) -> np.ndarray:
        """Embed a numerical column onto a randomly chosen great circle of the unit
        sphere in the embedding space.

        Args:
            column_data (np.ndarray): The numerical data for the column.
            col (str): Name of the column being processed.

        Returns:
            np.ndarray: Embedded values of shape (len(column_data), embed_dim).
        """
        point = self.column_properties[col]["point"].reshape([1, self.embed_dim])
        vec = self.column_properties[col]["vec"].reshape([1, self.embed_dim])
        col_min = self.column_properties[col]["min_value"]
        col_max = self.column_properties[col]["max_value"]
        col_range = col_max - col_min

        alpha = (np.pi * (0.8 * (column_data - col_min) / col_range + 0.1)).reshape(
            [len(column_data), 1]
        )
        embeddings = np.sin(alpha) * point + np.cos(alpha) * vec
        embeddings[np.argwhere(np.isnan(column_data))] = np.zeros(
            self.embed_dim, dtype=float
        )

        return np.array(embeddings, dtype=float)

    def _embed_categorical_column(self, column_data: pd.Series, col: str) -> np.ndarray:
        """
        Embed a categorical column into the embedding space.

        Maps each category to a point on the unit sphere.
        Unknown categories encountered during transform are assigned new points
        dynamically.

        Args:
            column_data (pd.Series): The categorical data for the column.
            col (str): Name of the column being processed.

        Returns:
            np.ndarray: Embedded values of shape (len(column_data), embed_dim).

        Raises:
            ValueError: If unique_category_embeddings is not a dictionary.
        """
        unique_category_embeddings = self.column_properties[col]

        if not isinstance(unique_category_embeddings, dict):
            raise ValueError(f"The unique category embedding is not an dictionary.")

        unique_categories = column_data.dropna().unique()
        for value in unique_categories:
            if value not in unique_category_embeddings.keys():
                unique_category_embeddings[value] = new_point_on_unit_sphere(
                    self.embed_dim
                )

        embeddings = np.empty((len(column_data), self.embed_dim), dtype=float)
        for i, value in enumerate(column_data):
            if pd.isna(value):
                embeddings[i, :] = np.zeros(self.embed_dim, dtype=float)
            else:
                embeddings[i, :] = unique_category_embeddings[value]

        return np.array(embeddings, dtype=float)


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
    return candidate_point / norm
