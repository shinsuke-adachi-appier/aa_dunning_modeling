"""
Calibrated model loading for production. Includes IsotonicCalibratedClassifier
so joblib can unpickle models trained with train_dunning_v2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


# Must be in scope for joblib.load to unpickle models saved from train_dunning_v2.
class IsotonicCalibratedClassifier:
    """Wraps a fitted classifier and calibrates probabilities via IsotonicRegression. Supports optional temperature scaling."""

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
            logit = np.log(p_cal / (1.0 - p_cal))
            p_cal = 1.0 / (1.0 + np.exp(-np.clip(logit / temperature, -500, 500)))
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.hstack([1 - p_cal, p_cal])

    @property
    def feature_names_(self):
        return getattr(self.estimator, "feature_names_", None)


def _register_for_unpickle():
    """Register this module's IsotonicCalibratedClassifier so joblib finds it."""
    import __main__
    setattr(__main__, "IsotonicCalibratedClassifier", IsotonicCalibratedClassifier)


def load_model(model_path: str) -> Any:
    """
    Load calibrated model from GCS (gs://...) or local path.
    Returns None on failure (caller should use fallback).
    """
    _register_for_unpickle()

    path = model_path
    if model_path.startswith("gs://"):
        try:
            from google.cloud import storage
            from urllib.parse import urlparse
            parsed = urlparse(model_path)
            bucket_name, blob_path = parsed.netloc, parsed.path.lstrip("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            local = Path("/tmp") / Path(blob_path).name
            blob.download_to_filename(str(local))
            path = str(local)
        except Exception as e:
            print(f"Failed to download model from GCS: {e}", file=sys.stderr)
            return None

    path_obj = Path(path)
    if not path_obj.exists():
        print(f"Model file not found: {path_obj}", file=sys.stderr)
        return None

    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        return None
