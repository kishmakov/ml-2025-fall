from sklearn.base import RegressorMixin
import numpy as np

class MeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        y_arr = np.asarray(y)
        self.mean_ = float(np.mean(y_arr))
        return self

    def predict(self, X=None):
        n_samples = np.asarray(X).shape[0]
        return np.full(shape=(n_samples,), fill_value=self.mean_, dtype=float)