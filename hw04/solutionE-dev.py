import numpy as np
import matplotlib.pyplot as plt

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

    def predict2(self, X):
        Xs = (X - self.x_mean_) / self.x_std_
        return Xs @ self.W_std_ + self.y_mean_

    def check_predict(self, X):
        X = np.asarray(X)

        p1 = self.predict(X)
        p2 = self.predict2(X)
        print(f"||p1 - p2||_2={np.linalg.norm(p1 - p2):.6f}")


def _read_csv_matrix(path):
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def _read_csv_vector(path):
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    return arr.reshape(-1)

if __name__ == "__main__":
    X = _read_csv_matrix("fx2.csv")
    Y = _read_csv_vector("fy2.csv")
    PX = _read_csv_matrix("px2.csv")

    model = SGDLinearRegressor(delta_converged=1e-03, max_steps=10000, batch_size=200)
    model.fit(X, Y)
    model.check_predict(PX)

    p1 = model.predict(X)
    p2 = model.predict2(X)

    print(f"||p1 - Y||_2={np.linalg.norm(p1 - Y):.6f}")
    print(f"||p2 - Y||_2={np.linalg.norm(p2 - Y):.6f}")

    ############################################ KS test

    # def _ks_two_sample_stat(a, b):
    #     a = np.asarray(a).reshape(-1)
    #     b = np.asarray(b).reshape(-1)
    #     a = a[np.isfinite(a)]
    #     b = b[np.isfinite(b)]
    #     if a.size == 0 or b.size == 0:
    #         return np.nan, a.size, b.size
    #     sa = np.sort(a)
    #     sb = np.sort(b)
    #     vals = np.unique(np.concatenate([sa, sb]))
    #     cdf_a = np.searchsorted(sa, vals, side="right") / sa.size
    #     cdf_b = np.searchsorted(sb, vals, side="right") / sb.size
    #     d = np.max(np.abs(cdf_a - cdf_b))
    #     return float(d), sa.size, sb.size

    # D, n_y, n_p = _ks_two_sample_stat(Y, preds)
    # crit_005 = 1.36 * np.sqrt((n_y + n_p) / (n_y * n_p)) if n_y > 0 and n_p > 0 else np.nan
    # same_dist = bool(D < crit_005) if np.isfinite(D) and np.isfinite(crit_005) else False

    # mean_y, mean_p = float(np.mean(Y)), float(np.mean(preds))
    # std_y, std_p = float(np.std(Y)), float(np.std(preds))
    # print(f"KS two-sample test (alpha=0.05): D={D:.6f}, crit={crit_005:.6f}, nY={n_y}, nPreds={n_p}, same_distribution={same_dist}")
    # print(f"Means: Y={mean_y:.6f}, preds={mean_p:.6f} | STDs: Y={std_y:.6f}, preds={std_p:.6f}")

    ############################################ plot

    # plt.figure(figsize=(8, 4))
    # combined = np.concatenate([preds.reshape(-1), Y.reshape(-1)])
    # bins = np.linspace(combined.min(), combined.max(), 50)

    # plt.hist(preds, bins=bins, alpha=0.6, label="preds", color="tab:blue", edgecolor="black")
    # plt.hist(Y, bins=bins, alpha=0.6, label="Y", color="tab:orange", edgecolor="black")

    # plt.xlabel("Value")
    # plt.ylabel("Count")
    # plt.title("Histogram of preds and Y")
    # plt.grid(True, linestyle=":", alpha=0.6)
    # plt.legend()
    # plt.tight_layout()
    # out_path = "preds.png"
    # plt.savefig(out_path, dpi=150)
    # print(f"Plot saved to {out_path}")
    # plt.close()
    # plt.savefig(out_path, dpi=150)
    # print(f"Plot saved to {out_path}")
    # plt.close()
