from abc import ABC, abstractmethod

from sklearn.base import TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


class OutlierDetectionAlgorithm(TransformerMixin, ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit_predict(self, X):
        pass


class LocalOutlierFactorAlgorithm(OutlierDetectionAlgorithm):
    def __init__(self, dim_reduction: bool = True, *args, **kwargs):
        super().__init__()
        self.local_outlier_algo = LocalOutlierFactor(*args, **kwargs)
        self.dim_reduction = dim_reduction

        if self.dim_reduction:
            raise NotImplementedError

    def fit(self, X):
        if self.dim_reduction:
            raise NotImplementedError
        else:
            self.local_outlier_algo.fit(X)

    def predict(self, X):
        if self.dim_reduction:
            raise NotImplementedError
        else:
            return self.local_outlier_algo.predict(X)

    def fit_predict(self, X):
        if self.dim_reduction:
            raise NotImplementedError
        else:
            return self.local_outlier_algo.fit_predict(X)


class IsolationForestAlgorithm(OutlierDetectionAlgorithm):
    def __init__(self, dim_reduction: bool = True, *args, **kwargs):
        super().__init__()
        self.isolation_forest_algo = IsolationForest(*args, **kwargs)
        self.dim_reduction = dim_reduction
        if self.dim_reduction:
            raise NotImplementedError

    def fit(self, X):
        if self.dim_reduction:
            raise NotImplementedError
        else:
            self.isolation_forest_algo.fit(X)

    def predict(self, X):
        if self.dim_reduction:
            raise NotImplementedError
        else:
            return self.isolation_forest_algo.predict(X)

    def fit_predict(self, X):
        if self.dim_reduction:
            raise NotImplementedError
        else:
            return self.isolation_forest_algo.fit_predict(X)
