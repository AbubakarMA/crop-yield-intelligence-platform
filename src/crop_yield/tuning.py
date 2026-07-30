"""Temporal hyperparameter-tuning utilities for crop-yield models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from crop_yield.evaluation import regression_metrics
from crop_yield.models import build_random_forest_pipeline
from crop_yield.preprocessing import split_features_target


@dataclass(frozen=True)
class TemporalFold:
    """One expanding-window training and validation period."""

    name: str
    train_end_year: int
    validation_start_year: int
    validation_end_year: int


@dataclass(frozen=True)
class RandomForestConfig:
    """Random-forest hyperparameters compared during tuning."""

    max_depth: int | None
    min_samples_leaf: int
    max_features: float


DEFAULT_TUNING_FOLDS = (
    TemporalFold("fold_1", 2001, 2002, 2003),
    TemporalFold("fold_2", 2003, 2004, 2005),
    TemporalFold("fold_3", 2005, 2006, 2007),
)


def expanding_window_partitions(
    data: pd.DataFrame,
    folds: Iterable[TemporalFold] = DEFAULT_TUNING_FOLDS,
    *,
    year_column: str = "year",
) -> list[tuple[TemporalFold, pd.DataFrame, pd.DataFrame]]:
    """Create expanding training windows followed by later validation years."""
    if year_column not in data.columns:
        raise ValueError(f"Missing year column: {year_column}")

    partitions = []
    for fold in folds:
        if fold.train_end_year >= fold.validation_start_year:
            raise ValueError(
                f"{fold.name} must validate strictly after its training years."
            )
        if fold.validation_start_year > fold.validation_end_year:
            raise ValueError(
                f"{fold.name} validation start cannot exceed its end."
            )

        training = data.loc[
            data[year_column].le(fold.train_end_year)
        ].copy()
        validation = data.loc[
            data[year_column].between(
                fold.validation_start_year,
                fold.validation_end_year,
            )
        ].copy()
        if training.empty or validation.empty:
            raise ValueError(f"{fold.name} created an empty partition.")

        partitions.append((fold, training, validation))

    return partitions


def evaluate_random_forest_configs(
    development_data: pd.DataFrame,
    configs: Iterable[RandomForestConfig],
    *,
    folds: Iterable[TemporalFold] = DEFAULT_TUNING_FOLDS,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank configurations by mean MAE across temporal validation folds."""
    partitions = expanding_window_partitions(development_data, folds)
    rows: list[dict[str, float | int | None]] = []

    for config in configs:
        row: dict[str, float | int | None] = {
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "max_features": config.max_features,
        }
        fold_maes = []

        for fold, training, validation in partitions:
            training_features, training_target = split_features_target(
                training
            )
            validation_features, validation_target = split_features_target(
                validation
            )
            model = build_random_forest_pipeline(
                n_estimators=n_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                random_state=random_state,
            )
            model.fit(training_features, training_target)
            predictions = model.predict(validation_features)
            fold_mae = regression_metrics(
                validation_target.to_numpy(),
                predictions,
            )["mae"]
            row[f"{fold.name}_mae"] = fold_mae
            fold_maes.append(fold_mae)

        row["mean_temporal_mae"] = sum(fold_maes) / len(fold_maes)
        row["worst_temporal_mae"] = max(fold_maes)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["mean_temporal_mae", "worst_temporal_mae"],
            ascending=True,
        )
        .reset_index(drop=True)
    )
