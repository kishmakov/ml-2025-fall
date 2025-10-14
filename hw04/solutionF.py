import numpy as np
import pandas as pd
from typing import Optional, List

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


interesting_columns = ["Overall_Qual", "Garage_Qual", "Sale_Condition", "MS_Zoning"]

class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]] = None):
        """
        If needed_columns is not None select only these columns.
        Keeps only numeric (continuous) columns and scales them.
        """
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()
        self.fitted_columns_ = None

    def _select_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        return data if self.needed_columns is None else data[self.needed_columns]

    def fit(self, data, *args):
        df = self._select_columns(data)
        numeric_df = df.select_dtypes(include=[np.number])
        self.fitted_columns_ = list(numeric_df.columns)
        self.scaler.fit(numeric_df.values)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if self.fitted_columns_ is None:
            raise RuntimeError("BaseDataPreprocessor is not fitted")
        df = self._select_columns(data)
        numeric_df = df[self.fitted_columns_]
        return self.scaler.transform(numeric_df.values)

class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, **kwargs):
        super(OneHotPreprocessor, self).__init__(**kwargs)
        self.categorical_columns = interesting_columns
        self.encoder = OneHotEncoder(handle_unknown="ignore", drop="first")
        self.cat_cols_ = None

    def fit(self, data, *args):
        super().fit(data, *args)
        df = self._select_columns(data)
        self.cat_cols_ = [c for c in self.categorical_columns if c in df.columns]
        self.encoder.fit(df[self.cat_cols_])
        return self

    def transform(self, data):
        if self.fitted_columns_ is None:
            raise RuntimeError("OneHotPreprocessor is not fitted")
        X_num = super().transform(data)
        X_cat = self.encoder.transform(data[self.cat_cols_]).toarray()
        return np.hstack([X_num, X_cat])


def make_ultimate_pipeline():
    return Pipeline([
        ("preprocess", OneHotPreprocessor()),
        ("regressor", Ridge(alpha=30.0, random_state=42))
    ])