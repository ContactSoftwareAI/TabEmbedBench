import logging

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor

from tabembedbench.embedding_models import AbstractEmbeddingGenerator
from tabembedbench.utils.config import EmbAggregation
from tabembedbench.utils.embedding_utils import compute_embeddings_aggregation
from tabembedbench.utils.preprocess_utils import infer_categorical_columns
from tabembedbench.utils.torch_utils import get_device

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger("TabPFN")


class UniversalTabPFNEmbedding(AbstractEmbeddingGenerator):
    def __init__(
        self,
        tabpfn_clf: TabPFNClassifier | None = None,
        tabpfn_reg: TabPFNRegressor | None = None,
        n_estimators: int = 1,
        estimator_agg_func: str | EmbAggregation = "first_element",
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators

        self.device = get_device()

        self._init_tabpfn_configs = {
            "device": self.device,
            "n_estimators": self.n_estimators,
            "ignore_pretraining_limits": True,
            "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
        }

        self.estimator_agg_func = estimator_agg_func

        if tabpfn_clf is None:
            tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        if tabpfn_reg is None:
            tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)

        self.cat_cols = None
        self.multi_class = None
        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg

    def _get_default_name(self) -> str:
        return "TabPFN"

    @property
    def task_only(self) -> bool:
        return False

    def preprocess_data(self, X: np.ndarray, train: bool = True):
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        if train:
            self.cat_cols = infer_categorical_columns(X)
        else:
            pass

        return X

    def _compute_embeddings_per_column(
        self,
        X: torch.tensor,
        agg_func: str | EmbAggregation = "mean",
    ):
        embs = []
        for column_idx in range(X.shape[1]):
            mask = torch.zeros_like(X).bool()
            mask[:, column_idx] = True
            X_train, y_train = (
                X[~mask].reshape(X.shape[0], -1),
                X[mask],
            )

            X_pred, _y_pred = X[~mask].reshape(X.shape[0], -1), X[mask]

            model = self.tabpfn_clf if column_idx in self.cat_cols else self.tabpfn_reg
            try:
                model.fit(X_train, y_train)
            except ValueError as e:
                if "Unknown label type: continuous" in str(e):
                    logger.info(
                        f"Column {column_idx} is continuous. Reverting to regression."
                    )
                    model = self.tabpfn_reg
                    model.fit(X_train, y_train)
                elif "exceeds the maximal number of classes" in str(e):
                    # TODO: Überlegen wie man kategorische Spalten mit mehr als 10 Elemente umgehen muss
                    logger.warning(
                        f"Column {column_idx} exceeds the maximal number of classes. Skipping this column as target."
                    )
                    continue
                else:
                    raise ValueError(e)

            if self.n_estimators > 1:
                estimator_embs = model.get_embeddings(X_pred)

                if self.estimator_agg_func == "first_element":
                    embs += [estimator_embs[0]]
                else:
                    embs += [
                        compute_embeddings_aggregation(
                            estimator_embs, self.estimator_agg_func
                        )
                    ]
            else:
                embs += [model.get_embeddings(X_pred)[0]]

        return compute_embeddings_aggregation(embs, agg_func)

    def _multiclass_codebook_reduction(self, X, y):
        raise NotImplementedError

    @staticmethod
    def _generate_codebook(
        num_estimators,
        num_classes: int,
        alphabet_size: int,
        random_state_instance: np.random.RandomState,
    ):
        raise NotImplementedError

    def compute_embeddings(
        self, X: torch.Tensor, agg_func: str | EmbAggregation = "mean"
    ) -> np.ndarray:
        embeddings = self._compute_embeddings_per_column(X)

        return embeddings

    def reset_embedding_model(self):
        self.tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        self.tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)
        self.cat_cols = None
        self.multi_class = None
