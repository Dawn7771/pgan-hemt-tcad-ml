"""Reproduce independent-test Vth region diagnostics from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent.parent / "pGaN_HEMT_TCAD_ML_Dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data)
    return parser.parse_args()


def region(vth: float) -> str:
    if vth < 0.7:
        return "low"
    if vth <= 3.0:
        return "mid"
    return "high"


def main() -> None:
    args = parse_args()
    source = args.data_dir / "04_reproducibility_metadata" / "vth_catboost_independent_test_predictions.csv"
    data = pd.read_csv(source, dtype={"sample_id": str})
    data["true_region"] = data["tcad_vth"].map(region)
    data["absolute_error_v"] = np.abs(data["pred_vth"] - data["tcad_vth"])

    region_metrics = {}
    for name in ("low", "mid", "high"):
        subset = data.loc[data["true_region"].eq(name)]
        region_metrics[name] = {
            "sample_count": int(len(subset)),
            "recognition_rate_percent": 100.0 * float((subset["pred_class"] == name).mean()),
            "mae_v": float(subset["absolute_error_v"].mean()),
        }

    result = {
        "test_sample_count": int(len(data)),
        "classification_accuracy_percent": 100.0 * float(accuracy_score(data["true_region"], data["pred_class"])),
        "r2": float(r2_score(data["tcad_vth"], data["pred_vth"])),
        "mae_v": float(mean_absolute_error(data["tcad_vth"], data["pred_vth"])),
        "region_metrics": region_metrics,
    }

    output_dir = Path("/results") if Path("/results").is_dir() else Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "vth_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
