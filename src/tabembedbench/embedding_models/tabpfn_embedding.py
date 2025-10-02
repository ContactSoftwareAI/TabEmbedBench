import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions.utils import infer_categorical_features

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

        self.tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        self.tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)
        self.unsupervised_model = TabPFNUnsupervisedModel(
            tabpfn_clf=self.tabpfn_clf, tabpfn_reg=self.tabpfn_reg
        )

        self._is_fitted = False

    @property
    def task_only(self) -> bool:
        return False

    def _preprocess_data(
            self,
            X: np.ndarray,
            train: bool = True,
            outlier: bool = False,
            **kwargs
    ):
        return torch.tensor(X, dtype=torch.float64)

    def _fit_model(
            self,
            X_preprocessed: torch.Tensor,
            categorical_indices: Optional[list[int]] = None,
            **kwargs
    ):
        if categorical_indices is not None:
            self.unsupervised_model.set_categorical_features(
                categorical_indices
            )
        else:
            cat_cols = infer_categorical_features(X_preprocessed)
            self.unsupervised_model.set_categorical_features(cat_cols)

        self._is_fitted = True

    def _compute_embeddings(
        self,
        X_preprocessed: torch.Tensor,
        **kwargs,
    ) -> np.ndarray:
        embs = self.unsupervised_model.get_embeddings_per_column(X_preprocessed)

    def reset_embedding_model(self):
        self.tabpfn_clf = TabPFNClassifier(**self._init_tabpfn_configs)
        self.tabpfn_reg = TabPFNRegressor(**self._init_tabpfn_configs)
        self.unsupervised_model = TabPFNUnsupervisedModel(
            tabpfn_clf=self.tabpfn_clf, tabpfn_reg=self.tabpfn_reg
        )
        self._is_fitted = False
