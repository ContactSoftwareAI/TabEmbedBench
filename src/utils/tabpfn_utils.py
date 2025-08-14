import numpy as np
import pandas as pd

from utils.preprocess_utils import infer_categorical_columns
from sklearn.model_selection import KFold
from tabpfn import TabPFNRegressor, TabPFNClassifier
from typing import Union, Tuple


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
        mixture: bool = True,
    ) -> Union[list[np.ndarray], Tuple[list[np.ndarray], list[np.ndarray]]]:
        """
        Computes embeddings for the provided dataset based on categorical and numeric columns.
        If the `mixture` flag is `True`, it computes combined embeddings for both categorical
        and numeric data together. Otherwise, it separates the embeddings for categorical
        and numeric data.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): Input data for which embeddings need to be
                computed. Can be a NumPy array or a Pandas DataFrame.
            cat_cols (list[Union[int, str]] | None): List of column indices or names identifying
                categorical columns. If None, the categorical columns will be inferred
                automatically from the input data.
            numeric_cols (list[Union[int, str]] | None): List of column indices or names
                identifying numeric columns. If None, the numeric columns will be automatically
                derived based on the categorical columns provided or inferred.
            mixture (bool): Flag indicating whether to compute mixed embeddings for both
                categorical and numeric data (`True`), or separate embeddings for each type
                (`False`).

        Returns:
            Union[list[np.ndarray], Tuple[list[np.ndarray], list[np.ndarray]]]: If `mixture` is
                `True`, returns a list of combined embeddings. If `mixture` is `False`, returns
                a tuple where the first element is a list of embeddings for categorical data and
                the second element is a list of embeddings for numeric data.

        Raises:
            NotImplementedError: If `numeric_cols` is None and no categorical columns are
                provided or inferred, making it impossible to automatically identify numeric
                columns.
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

        X = X.to_numpy()

        if mixture:
            return self._compute_embeddings(X, cat_cols_idx, numeric_cols_idx)
        else:
            cat_mask = np.ones(X.shape[-1], dtype=bool)
            cat_mask[numeric_cols_idx] = False
            num_mask = np.ones(X.shape[-1], dtype=bool)
            num_mask[cat_cols_idx] = False

            X_cat = X[:, cat_mask]
            X_num = X[:, num_mask]

            cat_embeddings = self._compute_embeddings(X_cat, range(X_cat.shape[-1]), [])
            num_embeddings = self._compute_embeddings(X_num, [], range(X_num.shape[-1]))

            return cat_embeddings, num_embeddings

    def _compute_embeddings(
        self,
        X: np.ndarray,
        cat_cols_idx: Union[list[int], range],
        numeric_cols_idx: Union[list[int], range],
    ) -> list[np.ndarray]:
        """
        Computes embeddings for each column in the dataset based on categorical or numeric
        classification, leveraging either classifier or regressor models. Handles datasets with
        either no folds or cross-validation using k-fold splits.

        Args:
            X (np.ndarray): The dataset array where each column represents a feature and
                each row represents a sample.
            cat_cols_idx (list[int]): The list of indices representing the categorical features
                in the dataset.
            numeric_cols_idx (list[int]): The list of indices representing the numeric features
                in the dataset.

        Returns:
            list[np.ndarray]: A list of numpy arrays where each element contains the embeddings
                for the corresponding feature in the dataset.
        """
        embeddings = []

        for col_idx in range(X.shape[-1]):
            target = X[:, col_idx]

            mask = np.ones(X.shape[-1], dtype=bool)
            mask[col_idx] = False

            tmp_embeddings = []
            if self.n_fold == 0:
                if col_idx in cat_cols_idx:
                    self.tabpfn_clf.fit(X[:, mask], target)
                    tmp_embeddings = self.tabpfn_clf.get_embeddings(
                        X[:, mask]
                    )
                elif col_idx in numeric_cols_idx:
                    self.tabpfn_reg.fit(X[:, mask], target)
                    tmp_embeddings = self.tabpfn_reg.get_embeddings(
                        X[:, mask]
                    )
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
                            self.tabpfn_clf.get_embeddings(
                                X_val_fold
                            )
                        )
                    elif col_idx in numeric_cols_idx:
                        self.tabpfn_reg.fit(X_train_fold, y_train_fold)

                        tmp_embeddings.append(
                            self.tabpfn_reg.get_embeddings(
                                X_val_fold, data_source="test"
                            )
                        )
                embeddings.append(np.concatenate(tmp_embeddings, axis=1))

        return embeddings