"""Isotonic-calibrated classifier (used by train and shadow; register on __main__ for joblib unpickle)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _logit(p: np.ndarray) -> np.ndarray:
    """Inverse sigmoid; clip p to (0, 1) to avoid inf."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))), 0.0, 1.0)


class IsotonicCalibratedClassifier:
    """
    Wraps a fitted classifier and calibrates probabilities via IsotonicRegression.
    Optional temperature scaling (temperature > 1) after isotonic reduces optimism
    by pulling probabilities toward 0.5.
    """

    def __init__(self, estimator, method: str = "isotonic", temperature: float = 1.0):
        self.estimator = estimator
        self.method = method
        self.temperature = float(temperature)
        self.calibrator_ = None

    def fit(self, X_cal, y_cal):
        p = self.estimator.predict_proba(X_cal)[:, 1]
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_.fit(p, y_cal)
        return self

    def predict_proba(self, X):
        p = self.estimator.predict_proba(X)[:, 1]
        p_cal = self.calibrator_.predict(p).reshape(-1, 1)
        p_cal = np.clip(p_cal, 1e-7, 1.0 - 1e-7)
        temperature = getattr(self, "temperature", 1.0)
        if temperature != 1.0 and temperature > 0:
            # Temperature scaling: dilate logits so probabilities are less extreme
            p_cal = _sigmoid(_logit(p_cal) / temperature)
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.hstack([1 - p_cal, p_cal])

    @property
    def feature_names_(self):
        return getattr(self.estimator, "feature_names_", None)
