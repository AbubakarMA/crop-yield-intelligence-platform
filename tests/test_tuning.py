"""Tests for leakage-safe temporal model tuning."""

import unittest

import pandas as pd

from crop_yield.tuning import (
    TemporalFold,
    expanding_window_partitions,
)


class TemporalTuningTests(unittest.TestCase):
    """Check that tuning folds preserve chronological order."""

    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "year": [1999, 2000, 2001, 2002, 2003, 2004],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )

    def test_expanding_fold_validates_after_training(self) -> None:
        fold = TemporalFold("example", 2001, 2002, 2003)

        _, training, validation = expanding_window_partitions(
            self.data,
            [fold],
        )[0]

        self.assertEqual(training["year"].tolist(), [1999, 2000, 2001])
        self.assertEqual(validation["year"].tolist(), [2002, 2003])
        self.assertLess(training["year"].max(), validation["year"].min())

    def test_overlapping_training_and_validation_is_rejected(self) -> None:
        fold = TemporalFold("bad", 2002, 2002, 2003)

        with self.assertRaisesRegex(ValueError, "strictly after"):
            expanding_window_partitions(self.data, [fold])

    def test_empty_validation_period_is_rejected(self) -> None:
        fold = TemporalFold("empty", 2003, 2005, 2006)

        with self.assertRaisesRegex(ValueError, "empty partition"):
            expanding_window_partitions(self.data, [fold])


if __name__ == "__main__":
    unittest.main()
