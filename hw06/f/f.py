import contextlib
import json
import os
import pathlib
import typing as tp
import uuid
import sklearn

import numpy as np
import pandas as pd
import typing as tp

from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score


class ExperimentHandler:
    """This class perfoms experiments with given model, measures metrics and logs everything for thorough comparison"""
    stacking_prediction_filename = 'cv_stacking_prediction.npy'
    test_stacking_prediction_filename = 'test_stacking_prediction.npy'


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
        X=self.X_train,
        y=self.y_train,
        scoring=metrics,
        cv=self._cv_iterable,
        n_jobs=self._n_jobs,
        return_train_score=False
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
    returns fitted estimator for convenience and les train overhead
    :param estimator: estimator to use
    :param run: run to log to
    :return: estimator fitted on train, stacking cross-val predictions, stacking test predictions
    """
    if hasattr(estimator, "predict_proba"):  # choose the most informative method for stacking predictions
        method = "predict_proba"
    elif hasattr(estimator, "decision_function"):
        method = "decision_function"
    else:
        method = "predict"
    cross_val_stacking_prediction = sklearn.model_selection.cross_val_predict(
        estimator,
        X=self.X_train,
        y=self.y_train,
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
