from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.utils.validation import check_is_fitted

class MostFrequentClassifier(ClassifierMixin):
    # Predicts the most frequent value (mode) from y_train
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
            raise ValueError("y must be provided for MostFrequentClassifier.fit")

        # Flatten y to 1D array
        y_arr = np.ravel(y)

        if y_arr.size == 0:
            raise ValueError("y must not be empty")

        # Compute the most frequent value; np.unique sorts labels, making ties deterministic
        labels, counts = np.unique(y_arr, return_counts=True)
        self.most_common_ = labels[np.argmax(counts)]

        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        check_is_fitted(self, attributes=["most_common_"])

        # If X is provided, replicate most_common_ for each sample; otherwise return single-element array
        if X is None:
            n = 1
        else:
            try:
                n = len(X)
            except TypeError:
                # Fall back if X has no len()
                n = getattr(X, 'shape', [1])[0] if getattr(X, 'shape', None) is not None else 1

        return np.full(shape=(n,), fill_value=self.most_common_)