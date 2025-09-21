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
        if y is None:
            raise ValueError("y must be provided for fitting")

        # Convert to numpy array for safety
        y_arr = np.asarray(y)
        if y_arr.size == 0:
            raise ValueError("y must not be empty")

        # Store learned parameter following sklearn's convention with trailing underscore
        self.mean_ = float(np.mean(y_arr))

        # Optional: store number of features seen if X is provided (not used in prediction)
        if X is not None:
            try:
                self.n_features_in_ = np.asarray(X).shape[1]
            except Exception:
                # Fallback when X is 1D or shape is not accessible; not critical for a mean regressor
                self.n_features_in_ = None

        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        if not hasattr(self, "mean_"):
            raise ValueError("This MeanRegressor instance is not fitted yet. Call 'fit' before using this estimator.")

        # Determine number of samples in X to match sklearn's API (returns shape (n_samples,))
        n_samples = 1 if X is None else np.asarray(X).shape[0]
        return np.full(shape=(n_samples,), fill_value=self.mean_, dtype=float)