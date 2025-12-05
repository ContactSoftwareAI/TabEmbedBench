import numpy as np
import polars as pl
import pandas as pd
from sklearn.base import TransformerMixin
from tabembedbench.sphere_model import SphereModelARF as SphereModel

from tabembedbench.embedding_models.abstractembedding import AbstractEmbeddingGenerator


class SphereBasedEmbedding(AbstractEmbeddingGenerator):
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
                    numeric_col_series = pd.to_numeric(col_series, errors='coerce').astype(float)

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
                        processed_cols[col_name] = col_series.replace([np.inf, -np.inf], np.nan)
                    elif pd.api.types.is_datetime64_any_dtype(dtype):
                        processed_cols[col_name] = (col_series.astype(np.int64) // 10 ** 9).astype(float)
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
                        processed_pd_dfs.append((col_series.astype(np.int64) // 10 ** 9).astype(float).to_frame())
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
            categorical_indices (list[int] | None, optional): Categorical column indices.
                Defaults to None.
            **kwargs: Additional keyword arguments (unused).
        """
        if categorical_column_names is None:
            categorical_column_names = []
            #categorical_indices = infer_categorical_columns(data)
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
        embeddings_test = None if X_test_preprocessed is None else self.model.transform(X_test_preprocessed)

        return embeddings_train, embeddings_test

    def _reset_embedding_model(self):
        """Reset the embedding model to its initial state.

        Clears all fitted column properties and metadata.
        """
        self.model = SphereModel(self.embed_dim)
