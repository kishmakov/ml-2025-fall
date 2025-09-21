from sklearn.base import RegressorMixin
import numpy as np

class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''

        y_arr = np.asarray(y)
        self.mean_ = float(np.mean(y_arr))

        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''

        n_samples = np.asarray(X).shape[0]
        return np.full(shape=(n_samples,), fill_value=self.mean_, dtype=float)