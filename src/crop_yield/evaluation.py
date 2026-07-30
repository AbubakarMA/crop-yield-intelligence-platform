"""Regression metrics and prediction diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate interpretable validation metrics for regression."""
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(root_mean_squared_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)),
    }


def prediction_diagnostics(predicted: np.ndarray) -> dict[str, float | int]:
    """Summarize physically impossible negative-yield predictions."""
    predicted_array = np.asarray(predicted, dtype=float)
    negative_count = int((predicted_array < 0).sum())
    return {
        "negative_prediction_count": negative_count,
        "negative_prediction_rate": float(
            negative_count / len(predicted_array)
        ),
        "minimum_prediction": float(predicted_array.min()),
        "maximum_prediction": float(predicted_array.max()),
    }
