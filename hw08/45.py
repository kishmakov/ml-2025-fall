import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.tree import DecisionTreeRegressor


class MyBinaryTreeGradientBoostingClassifier:
    """
    *Binary* gradient boosting with trees using
    negative log-likelihood loss with constant learning rate.
    Trees are to predict logits.
    """
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            seed: int,
            **kwargs
    ):
        """
        :param n_estimators: estimators count
        :param learning_rate: hard learning rate
        :param seed: global seed
        :param kwargs: kwargs of base estimator which is sklearn TreeRegressor
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_logits = None
        self.rng = np.random.default_rng(seed)
        self.base_estimator = DecisionTreeRegressor
        self.base_estimator_kwargs = kwargs
        self.estimators = []
        self.loss_history = []  # this is to track model learning process

    def create_new_estimator(self, seed):
        return self.base_estimator(**self.base_estimator_kwargs, random_state=seed)

    @staticmethod
    def cross_entropy_loss(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        """
        compute negative log-likelihood for logits,
        use clipping for logarithms with self.eps
        or use numerically stable special functions.
        This is used to track model learning process
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return:
        """

        probas = 1 / (1 + np.exp(-logits))
        y_pred = np.clip(probas, MyBinaryTreeGradientBoostingClassifier.eps,
                         1 - MyBinaryTreeGradientBoostingClassifier.eps)

        cross_entropy = true_labels * np.log(y_pred) + (1 - true_labels) * np.log(1 - y_pred)
        return -np.sum(cross_entropy)

    @staticmethod
    def cross_entropy_loss_gradient(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        """
        compute gradient of log-likelihood w.r.t logits,
        use clipping for logarithms with self.eps
        or use numerically stable special functions
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return:
        """
        y_pred = 1 / (1 + np.exp(-logits))
        return y_pred - true_labels

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        """
        sequentially fit estimators to reduce residual on each iteration
        :param X: [n_samples, n_features]
        :param y: [n_samples]
        :return: self
        """
        self.loss_history = []
        assert (np.unique(y) == np.arange(2)).all()
        # init predictions with mean target
        self.initial_logits = np.log(np.sum(y) / (y.shape[0] - np.sum(y)))
        # create starting logits
        logits = self.initial_logits
        # init loss history with starting negative log-likelihood
        self.loss_history.append(self.cross_entropy_loss(y, logits))
        # sequentially fit estimators with random seeds
        for seed in self.rng.choice(
                max(self.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            estimator = self.create_new_estimator(seed)
            self.estimators.append(estimator)
            # compute gradient
            gradient = self.cross_entropy_loss_gradient(true_labels=y, logits=logits)
            # fit estimator on gradient residual
            estimator.fit(X=X, y=gradient)
            # adjust logits with learning rate
            logits -= self.learning_rate * gradient
            self.loss_history.append(self.cross_entropy_loss(y, logits))

        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        """
        :param X: [n_samples]
        :return:
        """
        result = self.initial_logits
        for estimator in self.estimators:
            result += self.learning_rate * estimator.predict(X)

        return 1 / (1 + np.exp(result))

    def predict(
            self,
            X: np.ndarray
    ):
        """
        calculate predictions using predict_proba
        :param X: [n_samples]
        :return:
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
