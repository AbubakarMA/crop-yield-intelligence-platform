"""Leakage-safe feature preparation for crop-yield regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

TARGET_COLUMN = "yield_hg_per_ha"

CATEGORICAL_FEATURES = [
    "area",
    "item",
]

STANDARD_NUMERIC_FEATURES = [
    "year",
    "average_rainfall_mm_per_year",
    "average_temperature_c",
]

SKEWED_NUMERIC_FEATURES = [
    "pesticides_tonnes",
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + STANDARD_NUMERIC_FEATURES
    + SKEWED_NUMERIC_FEATURES
)


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and the crop-yield regression target."""
    required_columns = set(MODEL_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing modelling columns: {missing}")

    features = data.loc[:, MODEL_FEATURES].copy()
    target = data.loc[:, TARGET_COLUMN].copy()
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build an inspectable preprocessing pipeline fitted on training only."""
    categorical_pipeline = Pipeline(
        steps=[
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    standard_numeric_pipeline = Pipeline(
        steps=[("scale", StandardScaler())]
    )

    pesticide_pipeline = Pipeline(
        steps=[
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                standard_numeric_pipeline,
                STANDARD_NUMERIC_FEATURES,
            ),
            (
                "pesticides",
                pesticide_pipeline,
                SKEWED_NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor.set_output(transform="pandas")
