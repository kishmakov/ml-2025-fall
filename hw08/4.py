import contextlib
import inspect
import json
import os
import pathlib
import typing as tp
import uuid

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
        params = dict(self.base_estimator_kwargs)
        sig = inspect.signature(self.base_estimator.__init__)
        if 'random_state' in sig.parameters:
            params['random_state'] = int(seed)
        estimator = self.base_estimator(**params)
        return estimator

    @staticmethod
    def cross_entropy_loss(
            true_labels: np.ndarray,
            logits: np.ndarray
    ) -> float:
        """
        Binary cross-entropy (negative log-likelihood) given logits.
        loss_i = log(1 + exp(z_i)) - y_i * z_i
        Uses logaddexp for numerical stability.
        Returns mean loss.
        """
        y = MyBinaryTreeGradientBoostingClassifier._normalize_labels(true_labels)
        z = np.asarray(logits, dtype=float).ravel()
        # logaddexp(0, z) = log(1 + exp(z)) stable
        loss_vec = np.logaddexp(0.0, z) - y * z
        return float(loss_vec.mean())

    @staticmethod
    def cross_entropy_loss_gradient(
            true_labels: np.ndarray,
            logits: np.ndarray
    ) -> np.ndarray:
        """
        Gradient of mean negative log-likelihood w.r.t logits.
        grad_i = sigmoid(z_i) - y_i
        """
        y = MyBinaryTreeGradientBoostingClassifier._normalize_labels(true_labels)
        z = np.asarray(logits, dtype=float).ravel()
        prob = MyBinaryTreeGradientBoostingClassifier._stable_sigmoid(z)
        return prob - y


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
        # only should be fitted on datasets with binary target
        assert (np.unique(y) == np.arange(2)).all()
        # init predictions with mean target (mind that these are logits!)
        p0 = np.clip(y.mean(), self.eps, 1 - self.eps)
        self.initial_logits = float(np.log(p0 / (1.0 - p0)))
        # create starting logits
        logits = np.full(X.shape[0], self.initial_logits, dtype=float)
        # init loss history with starting negative log-likelihood
        self.loss_history.append(self.cross_entropy_loss(y, logits))
        # sequentially fit estimators with random seeds
        for seed in self.rng.choice(
                max(self.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            # add newly created estimator
            est = self.create_new_estimator(seed)
            self.estimators.append(est)
            # compute gradient
            gradient = self.cross_entropy_loss_gradient(y, logits)
            # fit estimator on gradient residual (negative gradient)
            residual = -gradient  # y - sigmoid(logits)
            est.fit(X, residual)
            # adjust logits with learning rate
            logits = logits + self.learning_rate * est.predict(X)
            # append new loss to history
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
        # init logits using precalculated values
        logits = np.full(X.shape[0], self.initial_logits if self.initial_logits is not None else 0.0, dtype=float)
        # sequentially adjust logits with learning rate
        for estimator in self.estimators:
            logits += self.learning_rate * estimator.predict(X)
        # don't forget to convert logits to probabilities
        p1 = 1.0 / (1.0 + np.exp(-logits))
        probas = np.column_stack([1.0 - p1, p1])
        return probas

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
        predictions = (probas[:, 1] >= 0.5).astype(int)
        return predictions

    @staticmethod
    def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[neg])
        out[neg] = exp_x / (1.0 + exp_x)
        return out

    @staticmethod
    def _normalize_labels(y: np.ndarray) -> np.ndarray:
        """Convert labels to {0,1} if they are {-1,1}."""
        y = y.astype(float).ravel()
        uniq = np.unique(y)
        if np.array_equal(uniq, np.array([-1.0, 1.0])):
            y = (y + 1.0) / 2.0  # map -1->0, 1->1
        return y
