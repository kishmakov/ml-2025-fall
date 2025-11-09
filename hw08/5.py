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
        return self.base_estimator(**self.base_estimator_kwargs, random_state=seed)

    @staticmethod
    def cross_entropy_loss(
            true_labels: np.ndarray,
            logits: np.ndarray
    ) -> float:
        """
         Numerically stable binary cross-entropy (negative log-likelihood) for logits.
        Supports targets in {0,1} or {-1,1}. Returns mean loss.
        {0,1}: L_i = log(1 + exp(z_i)) - y_i * z_i
        {-1,1}: L_i = log(1 + exp(-y_i * z_i))
        """
        y = np.asarray(true_labels, dtype=float).ravel()
        z = np.asarray(logits, dtype=float).ravel()
        p = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        p[neg] = ez / (1.0 + ez)
        eps = MyBinaryTreeGradientBoostingClassifier.eps
        p = np.clip(p, eps, 1.0 - eps)
        loss_vec = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(loss_vec.mean())

    @staticmethod
    def cross_entropy_loss_gradient(
            true_labels: np.ndarray,
            logits: np.ndarray
    ) -> np.ndarray:
        """
        Gradient of mean negative log-likelihood w.r.t logits.
        {0,1}: grad_i = sigmoid(z_i) - y_i
        {-1,1}: grad_i = -y_i * sigmoid(-y_i * z_i)
        """
        y = np.asarray(true_labels, dtype=float).ravel()
        z = np.asarray(logits, dtype=float).ravel()
        p = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        p[neg] = ez / (1.0 + ez)
        return p - y

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
        Return positive class probabilities as 1D array [n_samples].
        """
        logits = np.full(X.shape[0], self.initial_logits if self.initial_logits is not None else 0.0, dtype=float)
        for estimator in self.estimators:
            logits += self.learning_rate * estimator.predict(X)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return p1  # shape (n_samples,)

    def predict(
            self,
            X: np.ndarray
    ):
        """
        Predict class labels (0/1) using 0.5 threshold on positive class probability.
        """
        p1 = self.predict_proba(X)
        return (p1 >= 0.5).astype(int)

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
