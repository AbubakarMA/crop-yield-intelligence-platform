"""Model-interpretation utilities for crop-yield regression."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def grouped_permutation_importance(
    model: object,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure MAE increase after shuffling each original feature.

    Shuffling before preprocessing keeps all one-hot columns belonging to one
    categorical feature together. A large positive MAE increase means the
    fitted model relied strongly on that feature for these observations.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if len(features) != len(target):
        raise ValueError("Features and target must contain equal rows.")
    if n_repeats < 2:
        raise ValueError("n_repeats must be at least 2.")

    observed = np.asarray(target, dtype=float)
    baseline_predictions = np.asarray(model.predict(features), dtype=float)
    baseline_mae = float(mean_absolute_error(observed, baseline_predictions))
    random_generator = np.random.default_rng(random_state)
    rows = []

    for feature in features.columns:
        mae_increases = []
        for _ in range(n_repeats):
            permuted = features.copy()
            permuted[feature] = random_generator.permutation(
                permuted[feature].to_numpy()
            )
            permuted_predictions = np.asarray(
                model.predict(permuted),
                dtype=float,
            )
            permuted_mae = mean_absolute_error(
                observed,
                permuted_predictions,
            )
            mae_increases.append(float(permuted_mae - baseline_mae))

        rows.append(
            {
                "feature": feature,
                "baseline_mae": baseline_mae,
                "mae_increase_mean": float(np.mean(mae_increases)),
                "mae_increase_std": float(
                    np.std(mae_increases, ddof=1)
                ),
                "mae_increase_min": float(np.min(mae_increases)),
                "mae_increase_max": float(np.max(mae_increases)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mae_increase_mean", ascending=False)
        .reset_index(drop=True)
    )


def grouped_native_feature_importance(model: object) -> pd.DataFrame:
    """Aggregate fitted forest importance back to original feature groups."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()
    native_importances = estimator.feature_importances_

    if len(transformed_names) != len(native_importances):
        raise RuntimeError(
            "Transformed feature names and model importances do not align."
        )

    def original_group(transformed_name: str) -> str:
        if transformed_name.startswith("area_"):
            return "area"
        if transformed_name.startswith("item_"):
            return "item"
        return transformed_name

    importance_table = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "native_importance": native_importances,
        }
    )
    importance_table["feature"] = importance_table[
        "transformed_feature"
    ].map(original_group)

    return (
        importance_table.groupby("feature", as_index=False)[
            "native_importance"
        ]
        .sum()
        .sort_values("native_importance", ascending=False)
        .reset_index(drop=True)
    )


def prediction_sensitivity(
    model: object,
    observation: pd.DataFrame,
    *,
    feature: str,
    values: Iterable[float | int],
) -> pd.DataFrame:
    """Predict one observation repeatedly while changing one feature."""
    if not isinstance(observation, pd.DataFrame):
        raise TypeError("observation must be a pandas DataFrame.")
    if len(observation) != 1:
        raise ValueError("observation must contain exactly one row.")
    if feature not in observation.columns:
        raise ValueError(f"Missing sensitivity feature: {feature}")

    scenarios = []
    for value in values:
        scenario = observation.copy()
        scenario.loc[:, feature] = value
        scenarios.append(scenario)
    scenario_frame = pd.concat(scenarios, ignore_index=True)

    return pd.DataFrame(
        {
            feature: scenario_frame[feature].to_numpy(),
            "prediction": np.asarray(
                model.predict(scenario_frame),
                dtype=float,
            ),
        }
    )
