import copy
from typing import Union, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tabpfn import TabPFNClassifier, TabPFNRegressor
import torch

from tabembedbench.embedding_models.base import BaseEmbeddingGenerator
from tabembedbench.utils.config import EmbAggregation
from tabembedbench.utils.embedding_utils import compute_embeddings_aggregation
from tabembedbench.utils.preprocess_utils import infer_categorical_columns


class UniversalTabPFNEmbedding(BaseEmbeddingGenerator):
    """
    UniversalTabPFNEmbedding provides functionality to generate embeddings for tabular
    data using TabPFNClassifier and TabPFNRegressor models.

    This class is designed to process both categorical and numeric columns in a tabular
    dataset. It uses k-fold cross-validation to generate feature embeddings for each
    column by training models on the remaining columns. The embeddings can be used
    for downstream machine learning tasks such as classification or regression.

    Attributes:
        tabpfn_clf (TabPFNClassifier): The TabPFNClassifier instance used for generating
            embeddings for categorical columns.
        tabpfn_reg (TabPFNRegressor): The TabPFNRegressor instance used for generating
            embeddings for numeric columns.
        n_fold (int): The number of folds used in k-fold cross-validation for
            embedding generation.
    """

    @property
    def task_only(self) -> bool:
        return False

    def __init__(
        self, tabpfn_clf: Optional[TabPFNClassifier] = None, tabpfn_reg: Optional[TabPFNRegressor] = None,
    ) -> None:
        super().__init__()
        if tabpfn_clf is None:
            tabpfn_clf = TabPFNClassifier()
        if tabpfn_reg is None:
            tabpfn_reg = TabPFNRegressor()
        self.cat_cols = None
        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg

    def _get_default_name(self) -> str:
        return "TabPFN"

    def preprocess_data(self, X: np.ndarray, train: bool = True):
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        self.X_ = copy.deepcopy(X)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if train:
            pass
        else:
            pass
        return X

    def get_embeddings(
        self,
        X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        cat_cols: list[Union[int, str]] | None = None,
        numeric_cols: list[Union[int, str]] | None = None,
    ) -> list[np.ndarray]:
        """
        Generates embeddings for each column of the provided dataset using trained
        models. The method processes categorical and numeric columns separately
        based on specified indices or inferred column types.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input dataset, either as a NumPy array
                or a pandas DataFrame.
            cat_cols (list[Union[int, str]] | None): List of indices or names (if
                X is a DataFrame) representing categorical columns. If None,
                categorical columns will be inferred.
            numeric_cols (list[Union[int, str]] | None): List of indices or names (if
                X is a DataFrame) representing numeric columns. If None, numeric columns
                will be inferred based on categorical column indices.

        Returns:
            list[np.ndarray]: A list where each element is an array containing
            embeddings for a corresponding column of the dataset.
        """
        if cat_cols is None:
            cat_cols_idx = infer_categorical_columns(X)
        elif isinstance(cat_cols[0], str) and isinstance(X, pd.DataFrame):
            cat_cols_idx = [X.get_loc(col_name) for col_name in cat_cols]
        else:
            cat_cols_idx = cat_cols

        if numeric_cols is None:
            if cat_cols_idx is None:
                raise NotImplementedError
            else:
                numeric_cols_idx = [
                    idx for idx in range(X.shape[-1]) if idx not in cat_cols_idx
                ]
        elif isinstance(numeric_cols[0], str) and isinstance(X, pd.DataFrame):
            numeric_cols_idx = [X.get_loc(col_name) for col_name in numeric_cols]
        else:
            numeric_cols_idx = numeric_cols

        X = X.numpy()

        embeddings = []

        for col_idx in range(X.shape[-1]):
            target = X[:, col_idx]

            mask = np.ones(X.shape[-1], dtype=bool)
            mask[col_idx] = False
            tmp_embeddings = []

            if col_idx in cat_cols_idx:
                self.tabpfn_clf.fit(X[:, mask], target)
                tmp_embeddings = self.tabpfn_clf.get_embeddings(
                    X[:, mask], data_source="test"
                )
            elif col_idx in numeric_cols_idx:
                self.tabpfn_reg.fit(X[:, mask], target)
                tmp_embeddings = self.tabpfn_reg.get_embeddings(
                    X[:, mask], data_source="test"
                )
            embeddings.append(tmp_embeddings[0])


        return embeddings

    def compute_embeddings(
        self, X: np.ndarray, agg_func: Union[str, EmbAggregation] = "mean"
    ) -> np.ndarray:
        """
        Computes the embeddings by aggregating the extracted embeddings using the specified aggregation
        function.

        Args:
            X (np.ndarray): Input data for which embeddings are to be computed.
            agg_func (Union[str, EmbAggregation]): Aggregation function to use for combining the
                extracted embeddings. Accepted values are a string specifying the function name
                (e.g., "mean") or an instance of EmbAggregation.

        Returns:
            ndarray: The aggregated embedding vectors for the input data.
        """
        X_embed = self.get_embeddings(X)
        return compute_embeddings_aggregation(X_embed, agg_func)
