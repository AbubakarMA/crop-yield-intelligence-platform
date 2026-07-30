"""Tests for leakage-safe feature preprocessing."""

import unittest

import numpy as np
import pandas as pd

from crop_yield.preprocessing import (
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    split_features_target,
)


def sample_training_data() -> pd.DataFrame:
    """Return a small modelling table with categorical and numeric features."""
    return pd.DataFrame(
        {
            "area": ["Ghana", "Ghana", "Kenya"],
            "item": ["Maize", "Rice, paddy", "Maize"],
            "year": [2005, 2006, 2007],
            "average_rainfall_mm_per_year": [1187.0, 1187.0, 630.0],
            "pesticides_tonnes": [0.0, 100.0, 10_000.0],
            "average_temperature_c": [26.5, 26.7, 24.0],
            "yield_hg_per_ha": [17000, 19000, 22000],
        }
    )


class FeaturePreprocessingTests(unittest.TestCase):
    """Check feature selection, transformation, and unknown categories."""

    def test_split_features_target_separates_yield(self) -> None:
        features, target = split_features_target(sample_training_data())

        self.assertEqual(features.columns.tolist(), MODEL_FEATURES)
        self.assertEqual(target.name, TARGET_COLUMN)
        self.assertNotIn(TARGET_COLUMN, features.columns)

    def test_missing_modelling_column_is_rejected(self) -> None:
        source = sample_training_data().drop(columns="item")

        with self.assertRaisesRegex(ValueError, "Missing modelling columns"):
            split_features_target(source)

    def test_unknown_categories_transform_without_failure(self) -> None:
        training_features, _ = split_features_target(sample_training_data())
        future_features = training_features.iloc[[0]].copy()
        future_features.loc[:, "area"] = "Sudan"
        future_features.loc[:, "item"] = "Sorghum"

        preprocessor = build_preprocessor()
        transformed_training = preprocessor.fit_transform(training_features)
        transformed_future = preprocessor.transform(future_features)

        self.assertEqual(
            transformed_future.shape[1],
            transformed_training.shape[1],
        )
        self.assertNotIn("area_Sudan", transformed_future.columns)
        self.assertNotIn("item_Sorghum", transformed_future.columns)
        self.assertTrue(np.isfinite(transformed_future.to_numpy()).all())

    def test_log1p_pipeline_handles_zero_pesticide_use(self) -> None:
        training_features, _ = split_features_target(sample_training_data())

        transformed = build_preprocessor().fit_transform(training_features)

        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertIn("pesticides_tonnes", transformed.columns)


if __name__ == "__main__":
    unittest.main()
