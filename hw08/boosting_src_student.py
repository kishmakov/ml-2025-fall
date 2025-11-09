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
        create new base estimator with proper keywords
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
        z = np.asarray(logits, dtype=float).ravel()
        y = np.asarray(true_labels).ravel()
        uniq = np.unique(y)
        if np.array_equal(uniq, np.array([-1, 1])) or np.array_equal(uniq, np.array([-1.0, 1.0])):
            # y in {-1,1}
            loss_vec = np.logaddexp(0.0, -y * z)
        else:
            # convert anything else to {0,1}
            if uniq.min() < 0:
                y = (y + 1.0) / 2.0
            loss_vec = np.logaddexp(0.0, z) - y * z
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
        z = np.asarray(logits, dtype=float).ravel()
        y = np.asarray(true_labels).ravel()
        uniq = np.unique(y)
        if np.array_equal(uniq, np.array([-1, 1])) or np.array_equal(uniq, np.array([-1.0, 1.0])):
            # y in {-1,1}
            yz = -y * z
            prob = MyBinaryTreeGradientBoostingClassifier._stable_sigmoid(yz)
            grad = -y * prob
        else:
            if uniq.min() < 0:
                y = (y + 1.0) / 2.0
            prob = MyBinaryTreeGradientBoostingClassifier._stable_sigmoid(z)
            grad = prob - y
        return grad

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


class Logger:
    """Logger performs data management and stores scores and other relevant information"""

    def __init__(self, logs_path: tp.Union[str, os.PathLike]):
        self.path = pathlib.Path(logs_path)

        records = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith('.json'):
                    uuid = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            logged_data = json.load(f)
                            records.append(
                                {
                                    'id': uuid,
                                    **logged_data
                                }
                            )
                        except json.JSONDecodeError:
                            pass
        if records:
            self.leaderboard = pd.DataFrame.from_records(records, index='id')
        else:
            self.leaderboard = pd.DataFrame(index=pd.Index([], name='id'))

        self._current_run = None

    class Run:
        """Run incapsulates information for a particular entry of logged material. Each run is solitary experiment"""

        def __init__(self, name, storage, path):
            self.name = name
            self._storage = storage
            self._path = path
            self._storage.append(pd.Series(name=name))

        def log(self, key, value):
            self._storage.loc[self.name, key] = value

        def log_values(self, log_values: tp.Dict[str, tp.Any]):
            for key, value in log_values.items():
                self.log(key, value)

        def save_logs(self):
            with open(self._path / f'{self.name}.json', 'w+') as f:
                json.dump(self._storage.loc[self.name].to_dict(), f)

        def log_artifact(self, fname: str, writer: tp.Callable):
            with open(self._path / fname, 'wb+') as f:
                writer(f)

    @contextlib.contextmanager
    def run(self, name: tp.Optional[str] = None):
        if name is None:
            name = str(uuid.uuid4())
        # elif name in self.leaderboard.index:
        #     raise NameError("Run with given name already exists, name should be unique")
        else:
            name = name.replace(' ', '_')
        self._current_run = Logger.Run(name, self.leaderboard, self.path / name)
        os.makedirs(self.path / name, exist_ok=True)
        try:
            yield self._current_run
        finally:
            self._current_run.save_logs()


def load_predictions_dataframe(filename, column_prefix, index):
    with open(filename, 'rb') as file:
        data = np.load(file)
        dataframe = pd.DataFrame(data, columns=[f'{column_prefix}_{i}' for i in range(data.shape[1])],
                                 index=index)
        return dataframe


class ExperimentHandler:
    """This class perfoms experiments with given model, measures metrics and logs everything for thorough comparison"""
    stacking_prediction_filename = 'cv_stacking_prediction.npy'
    test_stacking_prediction_filename = 'test_stacking_prediction.npy'

    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            cv_iterable: tp.Iterable,
            logger: Logger,
            metrics: tp.Dict[str, tp.Union[tp.Callable, str]],
            n_jobs=-1
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._cv_iterable = cv_iterable
        self.logger = logger
        self._metrics = metrics
        self._n_jobs = n_jobs

    def score_test(self, estimator, metrics, run, test_data=None):
        """
        Computes scores for test data and logs them to given run
        :param estimator: fitted estimator
        :param metrics: metrics to compute
        :param run: run to log into
        :param test_data: optional argument if one wants to pass augmented test dataset
        :return: None
        """
        if test_data is None:
            test_data = self.X_test
        test_scores = _score(estimator, test_data, self.y_test, metrics)
        run.log_values({key + '_test': value for key, value in test_scores.items()})

    def score_cv(self, estimator, metrics, run):
        """
        computes scores on cross-validation
        :param estimator: estimator to fit
        :param metrics: metrics to compute
        :param run: run to log to
        :return: None
        """
        cross_val_results = sklearn.model_selection.cross_validate(
            estimator,
            self.X_train,
            self.y_train,
            cv=self._cv_iterable,
            n_jobs=self._n_jobs,
            scoring=metrics
        )
        for key, value in cross_val_results.items():
            if key.startswith('test_'):
                metric_name = key.split('_', maxsplit=1)[1]
                mean_score = np.mean(value)
                std_score = np.std(value)
                run.log_values(
                    {
                        metric_name + '_mean': mean_score,
                        metric_name + '_std': std_score
                    }
                )

    def generate_stacking_predictions(self, estimator, run):
        """
        generates predictions over cross-validation folds, then saves them as artifacts
        returns fitted estimator for convinience and les train overhead
        :param estimator: estimator to use
        :param run: run to log to
        :return: estimator fitted on train, stacking cross-val predictions, stacking test predictions
        """
        if hasattr(estimator, "predict_proba"):
            method = "predict_proba"
        elif hasattr(estimator, "decision_function"):
            method = "decision_function"
        else:
            method = "predict"
        cross_val_stacking_prediction = sklearn.model_selection.cross_val_predict(
            estimator,
            self.X_train,
            self.y_train,
            cv=self._cv_iterable,
            n_jobs=self._n_jobs,
            method=method
        )
        run.log_artifact(ExperimentHandler.stacking_prediction_filename,
                         lambda file: np.save(file, cross_val_stacking_prediction))
        estimator.fit(self.X_train, self.y_train)
        test_stacking_prediction = getattr(estimator, method)(self.X_test)
        run.log_artifact(ExperimentHandler.test_stacking_prediction_filename,
                         lambda file: np.save(file, test_stacking_prediction))
        return estimator, cross_val_stacking_prediction, test_stacking_prediction

    def get_metrics(self, estimator):
        """
        get callable metrics with estimator validation
        (e.g. estimator has predict_proba necessary for likelihood computation, etc)
        """
        return _check_multimetric_scoring(estimator, self._metrics)

    def run(self, estimator: sklearn.base.BaseEstimator, name=None):
        """
        perform run for given estimator
        :param estimator: estimator to use
        :param name: name of run for convinience and consitent logging
        :return: leaderboard with conducted run
        """
        metrics = self.get_metrics(estimator)
        with self.logger.run(name=name) as run:
            # compute predictions over cross-validation
            self.score_cv(estimator, metrics, run)
            fitted_on_train, _, _ = self.generate_stacking_predictions(estimator, run)
            self.score_test(fitted_on_train, metrics, run, test_data=self.X_test)
            return self.logger.leaderboard.loc[[run.name]]

    def get_stacking_predictions(self, run_names):
        """
        :param run_names: run names for which to extract stacking predictions for averaging and stacking
        :return: dataframe with predictions indexed by run names
        """
        train_dataframes = []
        test_dataframes = []
        for run_name in run_names:
            train_filename = self.logger.path / run_name / ExperimentHandler.stacking_prediction_filename
            train_dataframes.append(load_predictions_dataframe(filename=train_filename, column_prefix=run_name,
                                                               index=self.X_train.index))
            test_filename = self.logger.path / run_name / ExperimentHandler.test_stacking_prediction_filename
            test_dataframes.append(load_predictions_dataframe(filename=test_filename, column_prefix=run_name,
                                                              index=self.X_test.index))

        return pd.concat(train_dataframes, axis=1), pd.concat(test_dataframes, axis=1)
