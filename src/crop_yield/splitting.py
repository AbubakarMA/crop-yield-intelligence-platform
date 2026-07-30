"""Leakage-resistant dataset splitting for crop-yield modelling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Training, validation, and test partitions ordered through time."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_train_validation_test_split(
    data: pd.DataFrame,
    *,
    train_end_year: int = 2007,
    validation_end_year: int = 2010,
    year_column: str = "year",
) -> TemporalSplit:
    """Split observations chronologically without using future information.

    Training contains years up to ``train_end_year``. Validation contains the
    following years through ``validation_end_year``. Test contains every later
    year. Each returned partition is sorted deterministically.
    """
    if year_column not in data.columns:
        raise ValueError(f"Missing year column: {year_column}")
    if data.empty:
        raise ValueError("Cannot split an empty dataset.")
    if train_end_year >= validation_end_year:
        raise ValueError(
            "train_end_year must be earlier than validation_end_year."
        )
    if data[year_column].isna().any():
        raise ValueError("Year values cannot be missing.")

    sort_columns = [
        column
        for column in [year_column, "area", "item"]
        if column in data.columns
    ]
    ordered = data.sort_values(sort_columns).reset_index(drop=True)

    train = ordered.loc[
        ordered[year_column].le(train_end_year)
    ].reset_index(drop=True)
    validation = ordered.loc[
        ordered[year_column].gt(train_end_year)
        & ordered[year_column].le(validation_end_year)
    ].reset_index(drop=True)
    test = ordered.loc[
        ordered[year_column].gt(validation_end_year)
    ].reset_index(drop=True)

    partitions = {
        "training": train,
        "validation": validation,
        "test": test,
    }
    empty_partitions = [
        name for name, partition in partitions.items() if partition.empty
    ]
    if empty_partitions:
        names = ", ".join(empty_partitions)
        raise ValueError(f"Temporal split created empty partitions: {names}")

    if len(train) + len(validation) + len(test) != len(ordered):
        raise RuntimeError("Temporal split did not allocate every observation.")

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )
