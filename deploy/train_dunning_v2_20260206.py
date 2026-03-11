# Stub so joblib can unpickle models saved from repo's train_dunning_v2_20260206.
# Models reference this module name; we provide the same class here.
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibratedClassifier:
    """Same as in lib.model; required for joblib.load of calibrated model."""

    def __init__(self, estimator, method="isotonic", temperature=1.0):
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
            p_cal = 1.0 / (1.0 + np.exp(-np.clip(np.log(p_cal / (1.0 - p_cal)) / temperature, -500, 500)))
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.hstack([1 - p_cal, p_cal])

    @property
    def feature_names_(self):
        return getattr(self.estimator, "feature_names_", None)
