import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class RubricCityMedianClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.medians_ = {}

    def fit(self, X, Y):
        X_df = pd.DataFrame(X, columns=['modified_rubrics', 'city'])
        df = pd.concat([X_df, pd.Series(Y, name='mean_receipt')], axis=1)
        # Group by the specified columns and calculate the median for each group
        self.medians_ = df.groupby(['modified_rubrics', 'city'])['mean_receipt'].median().to_dict()

        return self

    def predict(self, X):
        predictions = []
        for _, row in pd.DataFrame(X, columns=['modified_rubrics', 'city']).iterrows():
            # Get the median from the pre-calculated dictionary
            key = (row['modified_rubrics'], row['city'])
            predictions.append(self.medians_.get(key, 0))

        return np.array(predictions)