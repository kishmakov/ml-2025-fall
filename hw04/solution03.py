import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import RegressorMixin

class ExponentialLinearRegression(RegressorMixin):
    def __init__(self, *args, **kwargs):
        self.ridge = Ridge(*args, **kwargs)
        self._fitted = False

    def fit(self, X, Y):
        Y = np.asarray(Y)
        if np.any(Y <= 0):
            raise ValueError("y must be positive for log-transform")
        self.ridge.fit(X, np.log(Y))
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("ExponentialLinearRegression is not fitted")
        return np.exp(self.ridge.predict(X))

    def get_params(self, deep=True):
        # expose Ridge params for GridSearchCV compatibility
        return self.ridge.get_params(deep=deep)

    def set_params(self, **params):
        self.ridge.set_params(**params)
        return self