from abc import ABC, abstractmethod

from sklearn.base import TransformerMixin

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

    def __call__(self, X):
        return self.fit_predict(X)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()


class LOF(OutlierDetectionAlgorithm):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass

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
