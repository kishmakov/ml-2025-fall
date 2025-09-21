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

class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        y_arr = np.asarray(y)
        cities = np.asarray(X['city'])

        # Compute per-city mean
        # Convert city labels to strings (msk/spb), keeping None/NaN as is for later handling
        cities_clean = np.asarray(cities)

        city_means = {}
        unique_cities = []
        for city in np.unique(cities_clean):
            if city is None or (isinstance(city, float) and np.isnan(city)):
                continue
            mask = (cities_clean == city)
            if np.any(mask):
                city_means[str(city)] = float(np.mean(y_arr[mask]))
                unique_cities.append(str(city))

        self.city_means_ = city_means
        self.cities_ = np.array(unique_cities, dtype=object)

        return self

    def predict(self, X=None):
        cities = np.asarray(X['city'])

        preds = np.empty(len(cities), dtype=float)
        for i, c in enumerate(cities):
            preds[i] = self.city_means_.get(str(c), 0)

        return preds