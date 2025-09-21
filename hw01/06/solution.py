from sklearn.base import ClassifierMixin
import numpy as np

class MostFrequentClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        y_arr = np.ravel(y)
        labels, counts = np.unique(y_arr, return_counts=True)
        self.most_common_ = labels[np.argmax(counts)]
        return self

    def predict(self, X=None):
        n_samples = np.asarray(X).shape[0]
        return np.full(shape=(n_samples,), fill_value=self.most_common_)