"""Data loading, validation, and cleaning for the crop-yield dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_REQUIRED_COLUMNS = {
    "Area",
    "Item",
    "Year",
    "hg/ha_yield",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
}

GRAIN_COLUMNS = ["area", "item", "year"]

OUTPUT_COLUMNS = [
    "area",
    "item",
    "year",
    "yield_hg_per_ha",
    "average_rainfall_mm_per_year",
    "pesticides_tonnes",
    "average_temperature_c",
]


def load_yield_data(path: str | Path) -> pd.DataFrame:
    """Load the merged Kaggle yield CSV."""
    return pd.read_csv(Path(path))


def validate_raw_schema(data: pd.DataFrame) -> None:
    """Raise a clear error when required source columns are unavailable."""
    missing_columns = RAW_REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")


def prepare_modeling_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return one validated observation per country, crop, and year.

    The published merged CSV contains repeated country-crop-year records caused
    by multiple temperature observations. Exact semantic duplicates are removed
    first so repeated source rows do not receive extra weight. Remaining
    temperatures are averaged within the intended observation grain.
    """
    validate_raw_schema(data)

    cleaned = data.drop(columns=["Unnamed: 0"], errors="ignore").copy()
    cleaned = cleaned.rename(
        columns={
            "Area": "area",
            "Item": "item",
            "Year": "year",
            "hg/ha_yield": "yield_hg_per_ha",
            "average_rain_fall_mm_per_year": (
                "average_rainfall_mm_per_year"
            ),
            "avg_temp": "average_temperature_c",
        }
    )

    cleaned["area"] = cleaned["area"].str.strip()
    cleaned["item"] = cleaned["item"].str.strip()
    cleaned = cleaned.drop_duplicates()

    invariant_columns = [
        "yield_hg_per_ha",
        "average_rainfall_mm_per_year",
        "pesticides_tonnes",
    ]
    conflicts = (
        cleaned.groupby(GRAIN_COLUMNS)[invariant_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise ValueError(
            "Conflicting yield, rainfall, or pesticide values exist within "
            f"{int(conflicts.sum())} country-crop-year groups."
        )

    modeling_data = (
        cleaned.groupby(GRAIN_COLUMNS, as_index=False)
        .agg(
            yield_hg_per_ha=("yield_hg_per_ha", "first"),
            average_rainfall_mm_per_year=(
                "average_rainfall_mm_per_year",
                "first",
            ),
            pesticides_tonnes=("pesticides_tonnes", "first"),
            average_temperature_c=("average_temperature_c", "mean"),
        )
        .loc[:, OUTPUT_COLUMNS]
        .sort_values(GRAIN_COLUMNS)
        .reset_index(drop=True)
    )

    validate_modeling_data(modeling_data)
    return modeling_data


def validate_modeling_data(data: pd.DataFrame) -> None:
    """Validate stable rules required before exploratory analysis or modelling."""
    if list(data.columns) != OUTPUT_COLUMNS:
        raise ValueError("The processed dataset has an unexpected schema.")
    if data.empty:
        raise ValueError("The processed dataset is empty.")
    if data.isna().any().any():
        raise ValueError("The processed dataset contains missing values.")
    if data.duplicated(GRAIN_COLUMNS).any():
        raise ValueError(
            "Country-crop-year observations are not unique."
        )
    if not data["year"].between(1990, 2013).all():
        raise ValueError("Year values fall outside the documented range.")
    if (data["yield_hg_per_ha"] <= 0).any():
        raise ValueError("Yield must be positive.")
    if (data["average_rainfall_mm_per_year"] <= 0).any():
        raise ValueError("Rainfall must be positive.")
    if (data["pesticides_tonnes"] < 0).any():
        raise ValueError("Pesticide use cannot be negative.")
    if not data["average_temperature_c"].between(-30, 50).all():
        raise ValueError("Temperature contains implausible values.")


def save_processed_data(data: pd.DataFrame, path: str | Path) -> Path:
    """Save validated modelling data and return the output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path
