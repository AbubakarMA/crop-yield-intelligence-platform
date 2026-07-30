"""Export the portable random forest to a compact browser-readable artifact."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import joblib
import numpy as np


def rounded(values: np.ndarray) -> list[float]:
    """Return compact float values without changing tree decisions materially."""
    return np.round(values.astype(float), 6).tolist()


def export_model(model_path: Path, metadata_path: Path, output_path: Path) -> None:
    """Serialize preprocessing state and forest trees as gzipped JSON."""
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    preprocessor = model.named_steps["preprocessor"]
    forest = model.named_steps["model"]

    one_hot = preprocessor.named_transformers_["categorical"].named_steps[
        "one_hot"
    ]
    numeric_scaler = preprocessor.named_transformers_["numeric"].named_steps[
        "scale"
    ]
    pesticide_scaler = preprocessor.named_transformers_[
        "pesticides"
    ].named_steps["scale"]

    payload = {
        "version": metadata["model_version"],
        "name": metadata["model_name"],
        "countries": one_hot.categories_[0].tolist(),
        "crops": one_hot.categories_[1].tolist(),
        "numeric_mean": rounded(numeric_scaler.mean_),
        "numeric_scale": rounded(numeric_scaler.scale_),
        "pesticide_mean": float(pesticide_scaler.mean_[0]),
        "pesticide_scale": float(pesticide_scaler.scale_[0]),
        "supported_years": metadata["supported_years"],
        "input_ranges": metadata["input_ranges"],
        "final_test": metadata["final_test"],
        "limitations": metadata["limitations"],
        "trees": [],
    }
    for estimator in forest.estimators_:
        tree = estimator.tree_
        payload["trees"].append(
            {
                "l": tree.children_left.tolist(),
                "r": tree.children_right.tolist(),
                "f": tree.feature.tolist(),
                "t": rounded(tree.threshold),
                "v": rounded(tree.value.reshape(-1)),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=9) as file:
        json.dump(payload, file, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="artifacts/crop_yield_random_forest.joblib",
        type=Path,
    )
    parser.add_argument(
        "--metadata",
        default="artifacts/model_metadata.json",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    export_model(arguments.model, arguments.metadata, arguments.output)
    print(
        f"Wrote {arguments.output} "
        f"({arguments.output.stat().st_size / 1_000_000:.2f} MB)"
    )


if __name__ == "__main__":
    main()
