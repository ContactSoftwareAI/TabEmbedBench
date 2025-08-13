import numpy as np
import pandas as pd

from embedding_utils import embeddings_aggregation
from utils.preprocess_utils import infer_categorical_columns
from sklearn.model_selection import Kfold
from tabpfn import TabPFNRegressor, TabPFNClassifier
from typing import Union


class UniversalTabPFNEmbedding:
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
        if self.model is None:
            raise ValueError("No model has been set.")

        if cat_cols is None:
            cat_cols_idx = infer_categorical_columns(X)
        elif isinstance(cat_cols[0], str) and isinstance(X, pd.DataFrame):
            cat_cols_idx = [
                X.get_loc(col_name) for col_name in cat_cols
            ]
        else:
            cat_cols_idx = cat_cols

        if numeric_cols is None:
            if cat_cols is None:
                raise NotImplementedError
            else:
                numeric_cols_idx = [
                    idx for idx in range(X.shape[-1]) if idx not in cat_cols
                ]
        elif isinstance(num_cols[0], str) and isinstance(X, pd.DataFrame):
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

            kf = Kfold(n_splits=self.n_fold, shuffle=False)
            tmp_embeddings = []
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
