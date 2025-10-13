import numpy as np

from sklearn.base import RegressorMixin

class SGDLinearRegressor(RegressorMixin):
    def __init__(
        self,
        lr = None,
        regularization = None,
        delta_converged = None,
        max_steps = None,
        batch_size = None,
    ):
        # print(f"init: lr={lr}, regularization={regularization}, delta_converged={delta_converged}, max_steps={max_steps}, batch_size={batch_size}")
        self.lr = 0.01 if lr is None else lr
        self.regularization = 0.0 if regularization is None else regularization
        self.max_steps = 5000 if max_steps is None else max_steps
        self.delta_converged = 1e-3 if delta_converged is None else delta_converged
        self.batch_size = 64 if batch_size is None else batch_size

        self._std_eps = 1e-3
        self.W = None
        self.b = None
        self.W_std_ = None

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1)

        # standardize features
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = np.maximum(X.std(axis=0), self._std_eps)
        Xs = (X - self.x_mean_) / self.x_std_

        # center target
        self.y_mean_ = Y.mean()
        Yc = Y - self.y_mean_

        n_samples, n_features = X.shape

        self.W = np.random.uniform(-1.0, 1.0, size=n_features).astype(float)
        prev_W = self.W.copy()

        delta = 1.0
        for step in range(self.max_steps):
            idx = np.random.randint(0, n_samples, size=self.batch_size)
            m = len(idx)

            Xb = Xs[idx]
            yb = Yc[idx]

            residual = Xb @ self.W - yb

            grad_W = (Xb.T @ residual) / m + self.regularization * self.W
            self.W = self.W - self.lr * grad_W

            delta = np.linalg.norm(self.W - prev_W) / max(1.0, np.linalg.norm(prev_W))
            if delta < self.delta_converged:
                break

            prev_W = self.W.copy()

        # print(f"fit: steps={step+1}, delta={delta:.6f}")

        self.W_std_ = self.W.copy()
        self.W = self.W / self.x_std_
        self.b = self.y_mean_ - self.x_mean_ @ self.W

        return self

    def predict(self, X):
        X = np.asarray(X)
        return (X @ self.W + self.b).astype(float)

