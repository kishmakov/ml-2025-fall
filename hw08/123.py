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


class MyAdaBoostClassifier:
    """
    Multiclass AdaBoost implementation with SAMME.R algorithm
    """
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            base_estimator: tp.Type[sklearn.base.BaseEstimator],
            seed: int,
            **kwargs
    ):
        """
        :param n_estimators: count of estimators
        :param base_estimator: base estimator (practically tree classifier)
        :param seed: global seed
        :param kwargs: keyword arguments of base estimator
        """
        self.n_classes = None
        self.error_history = []  # this is to track model learning process
        self.n_estimators = n_estimators
        self.rng = np.random.default_rng(seed)
        self.base_estimator = base_estimator
        self.base_estimator_kwargs = kwargs
        # deduce which keywords are used to set seed for an estimator (sklearn or own tree implementation)
        signature = inspect.signature(self.base_estimator.__init__)
        self.seed_keyword = None
        if 'seed' in signature.parameters:
            self.seed_keyword = 'seed'
        elif 'random_state' in signature.parameters:
            self.seed_keyword = 'random_state'
        self.estimators = []

    def create_new_estimator(
            self,
            seed: int
    ):
        """
        create new base estiamtor with proper keywords
        and new *unique* seed
        :param seed:
        :return:
        """
        params = dict(self.base_estimator_kwargs)
        if self.seed_keyword is not None:
            params[self.seed_keyword] = int(seed)
        estimator = self.base_estimator(**params)
        return estimator

    def get_new_weights(
            self,
            true_labels: np.ndarray,
            predictions: np.ndarray,
            weights: np.ndarray
    ):
        """
        Calculate new weights according to SAMME.R scheme
        :param true_labels: [n_samples]
        :param predictions: [n_samples, n_classes]
        :param weights:     [n_samples]
        :return: normalized weights for next estimator fitting
        """
        K = self.n_classes
        proba = np.clip(predictions, self.eps, 1.0)
        # Y coding: +1 for true class, -1/(K-1) otherwise
        Y = np.full_like(proba, fill_value=-1.0 / (K - 1))
        Y[np.arange(true_labels.shape[0]), true_labels] = 1.0
        factor = (K - 1.0) / K
        new_weights = weights * np.exp(-factor * (Y * np.log(proba)).sum(axis=1))
        s = new_weights.sum()
        if s <= 0:
            new_weights = np.full_like(new_weights, 1.0 / new_weights.size)
        else:
            new_weights /= s
        return new_weights

    @staticmethod
    def get_estimator_error(
            estimator: sklearn.base.BaseEstimator,
            X: np.ndarray,
            y: np.ndarray,
            weights: np.ndarray
    ):
        """
        calculate weighted error of an estimator
        :param estimator:
        :param X:       [n_samples, n_features]
        :param y:       [n_samples]
        :param weights: [n_samples]
        :return:
        """
        y_pred = estimator.predict(X)
        err = np.average(y_pred != y, weights=weights)
        return err

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        """
        sequentially fit estimators with updated weights on each iteration
        :param X: [n_samples, n_features]
        :param y: [n_samples]
        :return: self
        """
        self.error_history = []
        # compute number of classes for internal use
        classes = np.unique(y)
        self.classes_ = classes
        self.n_classes = classes.size
        # init weights uniformly over all samples
        n = X.shape[0]
        weights = np.full(n, 1.0 / n)
        # helper to get proba aligned to self.classes_
        def _predict_proba_aligned(est, X_):
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X_)
                # align to self.classes_
                p_aligned = np.full((X_.shape[0], self.n_classes), self.eps)
                if hasattr(est, "classes_"):
                    idx_map = {c: i for i, c in enumerate(est.classes_)}
                    for j, c in enumerate(self.classes_):
                        if c in idx_map:
                            p_aligned[:, j] = p[:, idx_map[c]]
                else:
                    p_aligned = p
                return np.clip(p_aligned, self.eps, 1.0)
            elif hasattr(est, "decision_function"):
                df = est.decision_function(X_)
                if df.ndim == 1:
                    # binary -> convert to two-class probs via sigmoid
                    p1 = 1.0 / (1.0 + np.exp(-df))
                    p = np.column_stack([1 - p1, p1])
                else:
                    # multiclass margins -> softmax
                    z = df - df.max(axis=1, keepdims=True)
                    ez = np.exp(z)
                    p = ez / ez.sum(axis=1, keepdims=True)
                # align if possible
                p_aligned = np.full((X_.shape[0], self.n_classes), self.eps)
                if hasattr(est, "classes_"):
                    idx_map = {c: i for i, c in enumerate(est.classes_)}
                    for j, c in enumerate(self.classes_):
                        if c in idx_map:
                            p_aligned[:, j] = p[:, idx_map[c]]
                else:
                    p_aligned = p
                return np.clip(p_aligned, self.eps, 1.0)
            else:
                # fallback: one-hot of predictions
                yhat = est.predict(X_)
                p = np.full((X_.shape[0], self.n_classes), self.eps)
                for j, c in enumerate(self.classes_):
                    p[:, j] = (yhat == c).astype(float)
                # smooth
                p = (p + self.eps)
                p /= p.sum(axis=1, keepdims=True)
                return p

        # sequentially fit each model and adjust weights
        for seed in self.rng.choice(
                max(MyAdaBoostClassifier.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            # add newly created estimator
            est = self.create_new_estimator(seed)
            self.estimators.append(est)
            # fit added estimator to data with current sample weights
            try:
                est.fit(X, y, sample_weight=weights)
            except TypeError:
                est.fit(X, y)
            # compute probability predictions
            proba = _predict_proba_aligned(est, X)
            # calculate weighted error of last estimator and append to error history
            self.error_history.append(self.get_estimator_error(est, X, y, weights))
            # compute new adjusted weights
            weights = self.get_new_weights(y, proba, weights)

        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        """
        predicts probability of each class
        :param X: [n_samples, n_features]
        :return: array of probabilities of a shape [n_samples, n_classes]
        """
        if not self.estimators:
            return np.full((X.shape[0], self.n_classes), 1.0 / self.n_classes)
        def _predict_proba_aligned(est, X_):
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X_)
            elif hasattr(est, "decision_function"):
                df = est.decision_function(X_)
                if df.ndim == 1:
                    p1 = 1.0 / (1.0 + np.exp(-df))
                    p = np.column_stack([1 - p1, p1])
                else:
                    z = df - df.max(axis=1, keepdims=True)
                    ez = np.exp(z)
                    p = ez / ez.sum(axis=1, keepdims=True)
            else:
                yhat = est.predict(X_)
                p = np.full((X_.shape[0], self.n_classes), self.eps)
                for j, c in enumerate(self.classes_):
                    p[:, j] = (yhat == c).astype(float)
                p = (p + self.eps)
                p /= p.sum(axis=1, keepdims=True)
            p_aligned = np.full((X_.shape[0], self.n_classes), self.eps)
            if hasattr(est, "classes_"):
                idx_map = {c: i for i, c in enumerate(est.classes_)}
                for j, c in enumerate(self.classes_):
                    if c in idx_map:
                        p_aligned[:, j] = p[:, idx_map[c]]
            else:
                p_aligned = p
            return np.clip(p_aligned, self.eps, 1.0)

        K = self.n_classes
        scores = np.zeros((X.shape[0], K))
        factor = (K - 1.0) / K
        for est in self.estimators:
            p = _predict_proba_aligned(est, X)
            scores += factor * np.log(np.clip(p, self.eps, 1.0))
        # softmax
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        probas = exp_s / exp_s.sum(axis=1, keepdims=True)
        return probas

    def predict(
            self,
            X: np.ndarray
    ):
        """
        predicts class (use predicted class probabilities)
        :param X: [n_samples, n_features]
        :return: array class predictions of a shape [n_samples]
        """
        probas = self.predict_proba(X)
        idx = np.argmax(probas, axis=1)
        predictions = self.classes_[idx]
        return predictions