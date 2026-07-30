"""Tests for leakage-resistant temporal splitting."""

import unittest

import pandas as pd

from crop_yield.splitting import temporal_train_validation_test_split


def sample_modeling_data() -> pd.DataFrame:
    """Return a small modelling table spanning all three partitions."""
    return pd.DataFrame(
        {
            "area": ["Ghana"] * 6,
            "item": ["Maize"] * 6,
            "year": [2006, 2007, 2008, 2010, 2011, 2013],
            "yield_hg_per_ha": [10, 11, 12, 13, 14, 15],
        }
    )


class TemporalSplittingTests(unittest.TestCase):
    """Check chronological boundaries and failure conditions."""

    def test_default_boundaries_allocate_expected_years(self) -> None:
        split = temporal_train_validation_test_split(sample_modeling_data())

        self.assertEqual(split.train["year"].tolist(), [2006, 2007])
        self.assertEqual(split.validation["year"].tolist(), [2008, 2010])
        self.assertEqual(split.test["year"].tolist(), [2011, 2013])

    def test_every_observation_is_allocated_once(self) -> None:
        source = sample_modeling_data()
        split = temporal_train_validation_test_split(source)

        allocated = pd.concat(
            [split.train, split.validation, split.test],
            ignore_index=True,
        )

        self.assertEqual(len(allocated), len(source))
        self.assertEqual(
            sorted(allocated["yield_hg_per_ha"]),
            sorted(source["yield_hg_per_ha"]),
        )

    def test_invalid_boundaries_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be earlier"):
            temporal_train_validation_test_split(
                sample_modeling_data(),
                train_end_year=2010,
                validation_end_year=2010,
            )

    def test_missing_year_column_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing year column"):
            temporal_train_validation_test_split(
                sample_modeling_data().drop(columns="year")
            )

    def test_empty_partition_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty partitions"):
            temporal_train_validation_test_split(
                sample_modeling_data(),
                train_end_year=2012,
                validation_end_year=2013,
            )


if __name__ == "__main__":
    unittest.main()
