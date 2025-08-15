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


class LOF(OutlierDetectionAlgorithm):
    def __init__(self, dim_reduction: bool = True, *args, **kwargs):
        self.local_outlier_algo = LocalOutlierFactor(*args, **kwargs)
        self.dim_reduction = dim_reduction

        if self.dim_reduction:
            raise NotImplementedError

    def fit(self, X):
        if self.dim_reduction:
            self.local_outlier_algo.fit(X)
        else:
            self.local_outlier_algo.fit(X)

    def predict(self, X):
        if self.dim_reduction:
            return self.local_outlier_algo.predict(X)
        else:
            return self.local_outlier_algo.predict(X)

    def fit_predict(self, X):
        pass


class IsolationForest(OutlierDetectionAlgorithm):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass

    def fit_predict(self, X):
        pass
