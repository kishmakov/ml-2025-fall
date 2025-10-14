import numpy as np
import pandas as pd

from typing import Optional, List
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

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