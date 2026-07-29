"""Tests for crop-yield data preparation."""

import unittest

import pandas as pd

from crop_yield.data import prepare_modeling_data, validate_raw_schema


def sample_raw_data() -> pd.DataFrame:
    """Return a small source-shaped dataset containing repeated temperatures."""
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "Area": ["Ghana", "Ghana", "Ghana"],
            "Item": ["Maize", "Maize", "Rice, paddy"],
            "Year": [2000, 2000, 2000],
            "hg/ha_yield": [17000, 17000, 19000],
            "average_rain_fall_mm_per_year": [1187.0, 1187.0, 1187.0],
            "pesticides_tonnes": [145.0, 145.0, 145.0],
            "avg_temp": [26.0, 28.0, 27.0],
        }
    )


class DataPreparationTests(unittest.TestCase):
    """Check the stable rules in the data-preparation pipeline."""

    def test_prepare_modeling_data_returns_unique_grain(self) -> None:
        result = prepare_modeling_data(sample_raw_data())

        self.assertEqual(len(result), 2)
        self.assertFalse(
            result.duplicated(["area", "item", "year"]).any()
        )

    def test_prepare_modeling_data_averages_repeated_temperatures(self) -> None:
        result = prepare_modeling_data(sample_raw_data())
        maize_temperature = result.loc[
            result["item"].eq("Maize"), "average_temperature_c"
        ].item()

        self.assertAlmostEqual(maize_temperature, 27.0)

    def test_prepare_modeling_data_removes_saved_index(self) -> None:
        result = prepare_modeling_data(sample_raw_data())

        self.assertNotIn("Unnamed: 0", result.columns)

    def test_validate_raw_schema_reports_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_raw_schema(pd.DataFrame({"Area": ["Ghana"]}))

    def test_conflicting_values_within_grain_are_rejected(self) -> None:
        source = sample_raw_data()
        source.loc[1, "hg/ha_yield"] = 99999

        with self.assertRaisesRegex(ValueError, "Conflicting yield"):
            prepare_modeling_data(source)


if __name__ == "__main__":
    unittest.main()
