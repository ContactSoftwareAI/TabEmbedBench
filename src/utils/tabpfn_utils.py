import numpy as np
import pandas as pd

from utils.preprocess_utils import infer_categorical_columns
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor, TabPFNClassifier
from typing import Union


class UniversalTabPFNEmbedding:
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
    def __init__(
        self,
        tabpfn_clf: TabPFNClassifier,
        tabpfn_reg: TabPFNRegressor,
        n_fold: int = 0,
    ) -> None:
        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg
        self.n_fold = n_fold

    def get_embeddings(
        self,
        X: Union[np.ndarray, pd.DataFrame],
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
            cat_cols_idx = [
                X.get_loc(col_name) for col_name in cat_cols
            ]
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
            numeric_cols_idx = [
                X.get_loc(col_name) for col_name in numeric_cols
            ]
        else:
            numeric_cols_idx = numeric_cols

        X = X.to_numpy()

        embeddings = []

        for col_idx in range(X.shape[-1]):
            target = X[:, col_idx]

            mask = np.ones(X.shape[-1], dtype=bool)
            mask[col_idx] = False
            tmp_embeddings = []
            if self.n_fold == 0:
                if col_idx in cat_cols_idx:
                    self.tabpfn_clf.fit(X[:, mask], target)
                    tmp_embeddings = self.tabpfn_clf.get_embeddings(X[:, mask], data_source="test")
                elif col_idx in numeric_cols_idx:
                    self.tabpfn_reg.fit(X[:, mask], target)
                    tmp_embeddings = self.tabpfn_reg.get_embeddings(X[:, mask], data_source="test")
                embeddings.append(tmp_embeddings)
            else:
                kf = KFold(n_splits=self.n_fold, shuffle=False)
                for train_idx, val_idx in kf.split(X):
                    X_train_fold = X[train_idx][:, mask]
                    X_val_fold = X[val_idx][:, mask]
                    y_train_fold = target[train_idx]

                    if col_idx in cat_cols_idx:
                        self.tabpfn_clf.fit(X_train_fold, y_train_fold)

                        tmp_embeddings.append(
                            self.tabpfn_clf.get_embeddings(X_val_fold, data_source="test")
                        )
                    elif col_idx in numeric_cols_idx:
                        self.tabpfn_reg.fit(X_train_fold, y_train_fold)

                        tmp_embeddings.append(
                            self.tabpfn_reg.get_embeddings(X_val_fold, data_source="test")
                        )
                embeddings.append(np.concatenate(tmp_embeddings, axis=1))

        return embeddings
