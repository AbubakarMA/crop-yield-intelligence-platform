"""Benchmark and baseline regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from crop_yield.preprocessing import build_preprocessor


class CropMedianRegressor(RegressorMixin, BaseEstimator):
    """Predict each crop's training median with a global fallback."""

    def __init__(self, crop_column: str = "item") -> None:
        self.crop_column = crop_column

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series | np.ndarray,
    ) -> "CropMedianRegressor":
        """Learn crop-specific and global medians from training data."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("CropMedianRegressor requires a pandas DataFrame.")
        if self.crop_column not in features.columns:
            raise ValueError(f"Missing crop column: {self.crop_column}")
        if len(features) != len(target):
            raise ValueError("Features and target must contain equal rows.")

        target_series = pd.Series(
            np.asarray(target),
            index=features.index,
            dtype=float,
        )
        training_pairs = pd.DataFrame(
            {
                "crop": features[self.crop_column].astype(str),
                "target": target_series,
            },
            index=features.index,
        )

        self.global_median_ = float(target_series.median())
        self.crop_medians_ = training_pairs.groupby("crop")[
            "target"
        ].median()
        self.n_features_in_ = features.shape[1]
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return crop medians and use the global median for unseen crops."""
        check_is_fitted(
            self,
            attributes=["global_median_", "crop_medians_"],
        )
        if not isinstance(features, pd.DataFrame):
            raise TypeError("CropMedianRegressor requires a pandas DataFrame.")
        if self.crop_column not in features.columns:
            raise ValueError(f"Missing crop column: {self.crop_column}")

        predictions = (
            features[self.crop_column]
            .astype(str)
            .map(self.crop_medians_)
            .fillna(self.global_median_)
        )
        return predictions.to_numpy(dtype=float)


def build_global_median_baseline() -> DummyRegressor:
    """Return a benchmark that predicts the training-target median."""
    return DummyRegressor(strategy="median")


def build_linear_regression_pipeline() -> Pipeline:
    """Return preprocessing and ordinary least-squares regression together."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )
