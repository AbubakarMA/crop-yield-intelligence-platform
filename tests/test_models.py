"""Tests for regression benchmarks, pipelines, and metrics."""

import unittest

import numpy as np
import pandas as pd

from crop_yield.evaluation import (
    prediction_diagnostics,
    regression_metrics,
)
from crop_yield.models import (
    CropMedianRegressor,
    build_global_median_baseline,
    build_linear_regression_pipeline,
    build_log_target_ridge_pipeline,
    build_poisson_gradient_boosting_pipeline,
    build_random_forest_pipeline,
    build_tuned_random_forest_pipeline,
)
from crop_yield.preprocessing import split_features_target


def sample_modeling_data() -> pd.DataFrame:
    """Return a small dataset suitable for every baseline model."""
    return pd.DataFrame(
        {
            "area": ["Ghana", "Ghana", "Kenya", "Kenya"],
            "item": ["Maize", "Maize", "Rice, paddy", "Rice, paddy"],
            "year": [2004, 2005, 2006, 2007],
            "average_rainfall_mm_per_year": [
                1187.0,
                1187.0,
                630.0,
                630.0,
            ],
            "pesticides_tonnes": [10.0, 12.0, 20.0, 25.0],
            "average_temperature_c": [26.5, 26.7, 24.0, 24.2],
            "yield_hg_per_ha": [16000, 18000, 22000, 26000],
        }
    )


class RegressionModelTests(unittest.TestCase):
    """Check benchmark behavior and reusable evaluation logic."""

    def test_global_median_baseline_predicts_one_training_value(self) -> None:
        features, target = split_features_target(sample_modeling_data())
        model = build_global_median_baseline().fit(features, target)

        predictions = model.predict(features)

        np.testing.assert_allclose(predictions, [20000.0] * 4)

    def test_crop_median_uses_global_fallback_for_unseen_crop(self) -> None:
        features, target = split_features_target(sample_modeling_data())
        model = CropMedianRegressor().fit(features, target)
        future = features.iloc[[0, 2]].copy()
        future.loc[future.index[1], "item"] = "Sorghum"

        predictions = model.predict(future)

        np.testing.assert_allclose(predictions, [17000.0, 20000.0])

    def test_linear_pipeline_tolerates_unseen_categories(self) -> None:
        features, target = split_features_target(sample_modeling_data())
        model = build_linear_regression_pipeline().fit(features, target)
        future = features.iloc[[0]].copy()
        future.loc[:, "area"] = "Sudan"
        future.loc[:, "item"] = "Sorghum"

        predictions = model.predict(future)

        self.assertEqual(predictions.shape, (1,))
        self.assertTrue(np.isfinite(predictions).all())

    def test_log_target_ridge_predictions_are_positive(self) -> None:
        features, target = split_features_target(sample_modeling_data())
        model = build_log_target_ridge_pipeline().fit(features, target)

        predictions = model.predict(features)

        self.assertTrue((predictions > 0).all())

    def test_random_forest_predictions_are_non_negative(self) -> None:
        features, target = split_features_target(sample_modeling_data())
        model = build_random_forest_pipeline().fit(features, target)

        predictions = model.predict(features)

        self.assertTrue((predictions >= 0).all())

    def test_random_forest_builder_applies_requested_hyperparameters(
        self,
    ) -> None:
        pipeline = build_random_forest_pipeline(
            n_estimators=25,
            max_depth=6,
            min_samples_leaf=3,
            max_features=0.5,
            random_state=7,
            n_jobs=1,
        )
        model = pipeline.named_steps["model"]

        self.assertEqual(model.n_estimators, 25)
        self.assertEqual(model.max_depth, 6)
        self.assertEqual(model.min_samples_leaf, 3)
        self.assertAlmostEqual(model.max_features, 0.5)
        self.assertEqual(model.random_state, 7)
        self.assertEqual(model.n_jobs, 1)

    def test_tuned_random_forest_uses_selected_configuration(self) -> None:
        model = build_tuned_random_forest_pipeline().named_steps["model"]

        self.assertEqual(model.n_estimators, 300)
        self.assertIsNone(model.max_depth)
        self.assertEqual(model.min_samples_leaf, 1)
        self.assertAlmostEqual(model.max_features, 0.8)

    def test_poisson_gradient_boosting_predictions_are_positive(self) -> None:
        features, target = split_features_target(sample_modeling_data())
        model = build_poisson_gradient_boosting_pipeline().fit(
            features,
            target,
        )

        predictions = model.predict(features)

        self.assertTrue((predictions > 0).all())

    def test_regression_metrics_match_known_example(self) -> None:
        metrics = regression_metrics(
            np.array([0.0, 2.0]),
            np.array([1.0, 1.0]),
        )

        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["r2"], 0.0)

    def test_prediction_diagnostics_count_negative_values(self) -> None:
        diagnostics = prediction_diagnostics(
            np.array([-3.0, 0.0, 8.0, -1.0])
        )

        self.assertEqual(diagnostics["negative_prediction_count"], 2)
        self.assertAlmostEqual(
            diagnostics["negative_prediction_rate"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
