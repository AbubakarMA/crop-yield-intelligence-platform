"""Tests for final evaluation and production-artifact behavior."""

from pathlib import Path

import pandas as pd

from crop_yield.production import (
    build_deployment_random_forest,
    load_production_artifacts,
    save_production_artifacts,
)


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": ["Ghana"] * 6 + ["Kenya"] * 6,
            "item": ["Maize"] * 3 + ["Rice, paddy"] * 3
            + ["Maize"] * 3 + ["Rice, paddy"] * 3,
            "year": [2005, 2006, 2007, 2005, 2006, 2007] * 2,
            "average_rainfall_mm_per_year": [1187.0] * 6 + [630.0] * 6,
            "pesticides_tonnes": list(range(10, 22)),
            "average_temperature_c": [26.0, 26.2, 26.1] * 4,
            "yield_hg_per_ha": list(range(15000, 27000, 1000)),
        }
    )


def test_deployment_builder_uses_portable_configuration() -> None:
    forest = build_deployment_random_forest().named_steps["model"]

    assert forest.n_estimators == 100
    assert forest.max_depth == 18
    assert forest.min_samples_leaf == 1


def test_artifact_round_trip(tmp_path: Path) -> None:
    data = sample_data()
    features = data.drop(columns=["yield_hg_per_ha"])
    target = data["yield_hg_per_ha"]
    model = build_deployment_random_forest().fit(features, target)
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "model_version": "test",
        "countries": ["Ghana", "Kenya"],
        "crops": ["Maize", "Rice, paddy"],
    }

    save_production_artifacts(
        model,
        metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    loaded_model, loaded_metadata = load_production_artifacts(
        model_path,
        metadata_path,
    )

    assert loaded_metadata == metadata
    assert loaded_model.predict(features.iloc[[0]]).shape == (1,)
