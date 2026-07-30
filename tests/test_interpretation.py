"""Tests for model-interpretation utilities."""

import unittest

import numpy as np
import pandas as pd

from crop_yield.interpretation import (
    grouped_permutation_importance,
    prediction_sensitivity,
)


class SignalEstimator:
    """Predict directly from the signal column for deterministic tests."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["signal"].to_numpy(dtype=float)


class YearEstimator:
    """Return a deterministic function of year."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["year"].to_numpy(dtype=float) * 2


class InterpretationTests(unittest.TestCase):
    """Check grouped importance and scenario sensitivity."""

    def test_permutation_importance_identifies_used_feature(self) -> None:
        features = pd.DataFrame(
            {
                "signal": [1.0, 2.0, 3.0, 4.0, 5.0],
                "noise": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        target = features["signal"].copy()

        importance = grouped_permutation_importance(
            SignalEstimator(),
            features,
            target,
            n_repeats=5,
            random_state=42,
        ).set_index("feature")

        self.assertGreater(
            importance.loc["signal", "mae_increase_mean"],
            0,
        )
        self.assertAlmostEqual(
            importance.loc["noise", "mae_increase_mean"],
            0,
        )

    def test_permutation_importance_requires_repeats(self) -> None:
        features = pd.DataFrame({"signal": [1.0, 2.0]})

        with self.assertRaisesRegex(ValueError, "at least 2"):
            grouped_permutation_importance(
                SignalEstimator(),
                features,
                features["signal"],
                n_repeats=1,
            )

    def test_prediction_sensitivity_changes_one_feature(self) -> None:
        observation = pd.DataFrame(
            {"year": [2008], "signal": [5.0]}
        )

        sensitivity = prediction_sensitivity(
            YearEstimator(),
            observation,
            feature="year",
            values=[2008, 2010],
        )

        self.assertEqual(sensitivity["year"].tolist(), [2008, 2010])
        self.assertEqual(
            sensitivity["prediction"].tolist(),
            [4016.0, 4020.0],
        )


if __name__ == "__main__":
    unittest.main()
