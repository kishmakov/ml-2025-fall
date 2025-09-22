import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class ModifiedFeaturesMedianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.medians_ = None
        self.global_median_ = None

    def fit(self, X, Y):
        if isinstance(X, pd.DataFrame):
            mf = X["modified_features"].reset_index(drop=True)
        elif isinstance(X, pd.Series):
            mf = X.reset_index(drop=True)
        else:
            mf = pd.Series(np.asarray(X).ravel(), name="modified_features")

        y = pd.Series(np.asarray(Y).ravel(), name="mean_receipt")

        df = pd.concat([mf, y], axis=1)

        self.medians_ = df.groupby("modified_features")["mean_receipt"].median().to_dict()
        self.global_median_ = float(y.median())

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            mf = X["modified_features"]
        elif isinstance(X, pd.Series):
            mf = X
        else:
            mf = pd.Series(np.asarray(X).ravel(), name="modified_features")

        preds = pd.Series(mf).map(self.medians_)
        preds = preds.fillna(self.global_median_).to_numpy(dtype=float)
        return preds