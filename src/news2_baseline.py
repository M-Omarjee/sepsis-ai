"""
NEWS2 Score as a Calibrated Baseline Predictor
==============================================

Wraps the UK National Early Warning Score 2 (NEWS2) as a sklearn-style
predictor so it can be benchmarked alongside ML models on equal footing.

Why this matters
----------------
The NEWS2 score is an ordinal risk category (0-20) — it was not designed
to produce calibrated probabilities. Evaluating it against ML models using
metrics that assume probabilities (e.g. Brier score) is therefore unfair.

This module fits an isotonic regression on training data to map NEWS2
scores onto well-calibrated probabilities. This is the standard
technique for evaluating ordinal scores as probabilistic predictors
(Niculescu-Mizil & Caruana, 2005).

Usage
-----
>>> from news2_baseline import NEWS2Baseline
>>> baseline = NEWS2Baseline()
>>> baseline.fit(X_train['news2_score'], y_train)
>>> probs = baseline.predict_proba(X_test['news2_score'])
"""

from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression


class NEWS2Baseline:
    """
    A minimal wrapper that treats NEWS2 score as a predictor and
    calibrates its outputs to probabilities via isotonic regression.

    Parameters
    ----------
    calibrate : bool, default=True
        If True, fit isotonic regression on training scores -> labels.
        If False, rescale NEWS2 to [0, 1] by dividing by max_score.
    max_score : int, default=20
        Maximum possible NEWS2 score. Used only when calibrate=False.
    """

    def __init__(self, calibrate: bool = True, max_score: int = 20):
        self.calibrate = calibrate
        self.max_score = max_score
        self._calibrator = None

    def fit(self, news2_scores, y):
        news2_scores = np.asarray(news2_scores, dtype=float)
        y = np.asarray(y, dtype=int)

        if self.calibrate:
            self._calibrator = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            )
            self._calibrator.fit(news2_scores, y)
        return self

    def predict_proba(self, news2_scores):
        """
        Return predicted probabilities in the sklearn [n_samples, 2] format.
        """
        news2_scores = np.asarray(news2_scores, dtype=float)

        if self.calibrate:
            if self._calibrator is None:
                raise RuntimeError("Call fit() before predict_proba().")
            p1 = self._calibrator.predict(news2_scores)
        else:
            p1 = np.clip(news2_scores / self.max_score, 0.0, 1.0)

        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, news2_scores, threshold=0.5):
        """Binary predictions at a given probability threshold."""
        return (self.predict_proba(news2_scores)[:, 1] >= threshold).astype(int)
