import numpy as np

from sklearn.base import RegressorMixin

class SGDLinearRegressor(RegressorMixin):
    def __init__(
        self,
        lr=0.01,
        regularization=1.,
        delta_converged=1e-2,
        max_steps=1000,
        batch_size=64,
    ):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

        # feature/target scaling params
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self._std_eps = 1e-12

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1)

        # standardize features, center target
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0, ddof=0)
        self.x_std_ = np.where(self.x_std_ < self._std_eps, 1.0, self.x_std_)
        Xn = (X - self.x_mean_) / self.x_std_
        self.y_mean_ = Y.mean()
        Yc = Y - self.y_mean_

        n_samples, n_features = Xn.shape
        if self.W is None:
            self.W = np.zeros(n_features, dtype=float)
            self.b = 0.0

        prev_W = self.W.copy()
        prev_b = float(self.b)

        for step in range(self.max_steps):
            idx = np.random.randint(0, n_samples, size=min(self.batch_size, n_samples))
            Xb = Xn[idx]
            yb = Yc[idx]

            pred = Xb @ self.W + self.b
            residual = pred - yb
            m = Xb.shape[0]

            grad_W = (2.0 / m) * (Xb.T @ residual) + 2.0 * self.regularization * self.W
            # regularize bias too (objective penalizes full theta)
            grad_b = (2.0 / m) * np.sum(residual)

            self.W = self.W - self.lr * grad_W
            self.b = self.b - self.lr * grad_b

            delta_W = self.W - prev_W
            delta_b = self.b - prev_b
            if np.sqrt(np.sum(delta_W ** 2) + (delta_b ** 2)) < self.delta_converged:
                break
            prev_W = self.W.copy()
            prev_b = float(self.b)

        return self

    def predict(self, X):
        if self.W is None or self.x_mean_ is None or self.x_std_ is None or self.y_mean_ is None:
            raise RuntimeError("SGDLinearRegressor is not fitted")
        X = np.asarray(X)
        Xn = (X - self.x_mean_) / self.x_std_
        y_c = Xn @ self.W + self.b
        return (y_c + self.y_mean_).astype(float)