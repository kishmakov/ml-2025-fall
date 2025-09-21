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

class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        """
        Learn mean target per city from training data.

        Parameters
        ----------
        X : array-like or pandas DataFrame, shape = (n_samples, n_features)
            Training features. Must contain a 'city' column or be a 1-D array-like of city labels.
        y : array-like, shape = (n_samples,)
            Target values.
        """
        if y is None:
            raise ValueError("y must be provided for fitting")

        y_arr = np.asarray(y)
        if y_arr.size == 0:
            raise ValueError("y must not be empty")

        if X is None:
            raise ValueError("X with 'city' must be provided for CityMeanRegressor")

        # Extract city column from X. Support pandas DataFrame/Series and numpy/py lists.
        cities = None
        # pandas DataFrame/Series path (duck-typed)
        if hasattr(X, 'ndim') and getattr(X, 'ndim') == 1 and not hasattr(X, 'columns'):
            # 1-D Series-like
            cities = np.asarray(X)
        elif hasattr(X, 'columns') and 'city' in getattr(X, 'columns'):
            # DataFrame with 'city' column
            cities = np.asarray(X['city'])
        else:
            X_arr = np.asarray(X)
            # If 1-D, assume it's directly the city labels
            if X_arr.ndim == 1:
                cities = X_arr
            else:
                # Try to find a column named 'city' via structured array or fail
                if isinstance(X, np.ndarray) and X.dtype.names and 'city' in X.dtype.names:
                    cities = np.asarray(X['city'])
                else:
                    raise ValueError("X must be a 1-D array/Series of city labels or a DataFrame with a 'city' column")

        if len(cities) != len(y_arr):
            raise ValueError("X and y must have the same number of samples")

        # Compute per-city mean
        # Convert city labels to strings (msk/spb), keeping None/NaN as is for later handling
        cities_clean = np.asarray(cities)
        # Build mapping city -> mean(y)
        city_means = {}
        unique_cities = []
        for city in np.unique(cities_clean):
            if city is None or (isinstance(city, float) and np.isnan(city)):
                continue
            mask = (cities_clean == city)
            if np.any(mask):
                city_means[str(city)] = float(np.mean(y_arr[mask]))
                unique_cities.append(str(city))

        # Global mean fallback
        self.global_mean_ = float(np.mean(y_arr))
        self.city_means_ = city_means
        self.cities_ = np.array(unique_cities, dtype=object)

        # Optional: store number of features seen (if DataFrame, count columns)
        try:
            self.n_features_in_ = np.asarray(X).shape[1]
        except Exception:
            self.n_features_in_ = None

        return self

    def predict(self, X=None):
        """
        Predict by returning the mean for the example's city.
        Falls back to global mean when city is unknown or missing.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            Must contain a 'city' column or be a 1-D array-like of city labels.
        """
        if not hasattr(self, 'city_means_') or not hasattr(self, 'global_mean_'):
            raise ValueError("This CityMeanRegressor instance is not fitted yet. Call 'fit' before using this estimator.")

        if X is None:
            # Return scalar as array of size 1 using global mean if no X is given
            return np.array([self.global_mean_], dtype=float)

        # Extract city labels from X
        if hasattr(X, 'ndim') and getattr(X, 'ndim') == 1 and not hasattr(X, 'columns'):
            cities = np.asarray(X)
        elif hasattr(X, 'columns') and 'city' in getattr(X, 'columns'):
            cities = np.asarray(X['city'])
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                cities = X_arr
            else:
                if isinstance(X, np.ndarray) and X.dtype.names and 'city' in X.dtype.names:
                    cities = np.asarray(X['city'])
                else:
                    raise ValueError("X must be a 1-D array/Series of city labels or a DataFrame with a 'city' column")

        # Map each city to its mean with fallback
        preds = np.empty(len(cities), dtype=float)
        for i, c in enumerate(cities):
            # Treat None/NaN as missing -> use global mean
            if c is None or (isinstance(c, float) and np.isnan(c)):
                preds[i] = self.global_mean_
            else:
                preds[i] = self.city_means_.get(str(c), self.global_mean_)

        return preds