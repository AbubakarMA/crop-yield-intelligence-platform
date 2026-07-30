"""Final evaluation, artifact creation, and production-model utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from crop_yield.data import validate_modeling_data
from crop_yield.evaluation import prediction_diagnostics, regression_metrics
from crop_yield.models import (
    build_random_forest_pipeline,
    build_tuned_random_forest_pipeline,
)
from crop_yield.preprocessing import MODEL_FEATURES, split_features_target
from crop_yield.splitting import temporal_train_validation_test_split

MODEL_VERSION = "1.0.0"
RESEARCH_MODEL_NAME = "random_forest_300_unbounded"
DEPLOYMENT_MODEL_NAME = "random_forest_100_depth_18"
TEST_PERIOD = "2011-2013"


@dataclass(frozen=True)
class FinalEvaluation:
    """Locked test-set results suitable for a model card."""

    model_name: str
    training_period: str
    test_period: str
    train_rows: int
    test_rows: int
    mae_hg_per_ha: float
    rmse_hg_per_ha: float
    r2: float
    negative_predictions: int


def build_deployment_random_forest():
    """Return the smaller forest used by the browser deployment.

    It was selected on the 2008-2010 validation period. The depth cap makes
    model transfer to a browser practical while preserving most of the tuned
    forest's validation accuracy.
    """
    return build_random_forest_pipeline(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )


def evaluate_locked_model(
    data: pd.DataFrame,
    *,
    deployment_optimized: bool = False,
) -> tuple[Any, FinalEvaluation, pd.DataFrame]:
    """Fit on 1990-2010 and evaluate once on the sealed 2011-2013 test set."""
    validate_modeling_data(data)
    split = temporal_train_validation_test_split(data)
    development = pd.concat(
        [split.train, split.validation],
        ignore_index=True,
    )
    development_features, development_target = split_features_target(
        development
    )
    test_features, test_target = split_features_target(split.test)
    model = (
        build_deployment_random_forest()
        if deployment_optimized
        else build_tuned_random_forest_pipeline()
    )
    model.fit(development_features, development_target)
    predictions = model.predict(test_features)
    metrics = regression_metrics(test_target.to_numpy(), predictions)
    diagnostics = prediction_diagnostics(predictions)

    result = FinalEvaluation(
        model_name=(
            DEPLOYMENT_MODEL_NAME
            if deployment_optimized
            else RESEARCH_MODEL_NAME
        ),
        training_period="1990-2010",
        test_period=TEST_PERIOD,
        train_rows=len(development),
        test_rows=len(split.test),
        mae_hg_per_ha=metrics["mae"],
        rmse_hg_per_ha=metrics["rmse"],
        r2=metrics["r2"],
        negative_predictions=int(
            diagnostics["negative_prediction_count"]
        ),
    )
    predictions_frame = split.test.loc[
        :,
        ["area", "item", "year", "yield_hg_per_ha"],
    ].copy()
    predictions_frame["prediction_hg_per_ha"] = predictions
    predictions_frame["absolute_error_hg_per_ha"] = (
        predictions_frame["yield_hg_per_ha"]
        - predictions_frame["prediction_hg_per_ha"]
    ).abs()
    return model, result, predictions_frame


def fit_full_deployment_model(data: pd.DataFrame):
    """Refit the deployment forest on all validated historical observations."""
    validate_modeling_data(data)
    features, target = split_features_target(data)
    return build_deployment_random_forest().fit(features, target)


def build_model_metadata(
    data: pd.DataFrame,
    *,
    final_evaluation: FinalEvaluation,
) -> dict[str, Any]:
    """Create deployment metadata and defensible input boundaries."""
    return {
        "model_version": MODEL_VERSION,
        "model_name": DEPLOYMENT_MODEL_NAME,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target": "yield_hg_per_ha",
        "target_display_unit": "tonnes per hectare",
        "features": MODEL_FEATURES,
        "supported_years": [
            int(data["year"].min()),
            int(data["year"].max()),
        ],
        "countries": sorted(data["area"].unique().tolist()),
        "crops": sorted(data["item"].unique().tolist()),
        "input_ranges": {
            column: [
                float(data[column].min()),
                float(data[column].max()),
            ]
            for column in [
                "average_rainfall_mm_per_year",
                "pesticides_tonnes",
                "average_temperature_c",
            ]
        },
        "final_test": asdict(final_evaluation),
        "limitations": [
            "The model estimates historical conditional yield; it is not a "
            "long-range climate or policy forecasting model.",
            "Rainfall is constant through time within each country in the "
            "source data.",
            "Pesticide use is a national total rather than a per-hectare rate.",
            "Feature importance is predictive, not causal.",
        ],
    }


def save_production_artifacts(
    model: Any,
    metadata: dict[str, Any],
    *,
    model_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Path, Path]:
    """Persist a fitted model and its reader-facing metadata."""
    destination = Path(model_path)
    metadata_destination = Path(metadata_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination, compress=3)
    metadata_destination.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return destination, metadata_destination


def load_production_artifacts(
    model_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load the trusted local model artifact and metadata."""
    model = joblib.load(Path(model_path))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return model, metadata


def run_training(
    *,
    data_path: str | Path,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Run final evaluation, refit on all history, and save artifacts."""
    data = pd.read_csv(Path(data_path))
    _, research_result, research_predictions = evaluate_locked_model(data)
    _, deployment_result, deployment_predictions = evaluate_locked_model(
        data,
        deployment_optimized=True,
    )
    deployment_model = fit_full_deployment_model(data)
    metadata = build_model_metadata(
        data,
        final_evaluation=deployment_result,
    )
    metadata["research_model_final_test"] = asdict(research_result)

    output_dir = Path(artifact_dir)
    model_path, metadata_path = save_production_artifacts(
        deployment_model,
        metadata,
        model_path=output_dir / "crop_yield_random_forest.joblib",
        metadata_path=output_dir / "model_metadata.json",
    )
    research_predictions.to_csv(
        output_dir / "research_model_test_predictions.csv",
        index=False,
    )
    deployment_predictions.to_csv(
        output_dir / "deployment_model_test_predictions.csv",
        index=False,
    )
    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "research_model_final_test": asdict(research_result),
        "deployment_model_final_test": asdict(deployment_result),
    }


def main() -> None:
    """Command-line entry point for reproducible artifact creation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/crop_yield_modeling.csv",
    )
    parser.add_argument("--output", default="artifacts")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_training(
                data_path=arguments.data,
                artifact_dir=arguments.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
