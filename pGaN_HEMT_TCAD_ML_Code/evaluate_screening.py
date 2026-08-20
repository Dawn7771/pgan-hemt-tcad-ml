"""Reproduce conditional candidate-screening metrics from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent.parent / "pGaN_HEMT_TCAD_ML_Dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data)
    return parser.parse_args()


def topk_retention(data: pd.DataFrame, k: int) -> float:
    predicted = set(data.nsmallest(k, "pred_rank")["sample_id"])
    tcad = set(data.nsmallest(k, "tcad_rank")["sample_id"])
    return 100.0 * len(predicted & tcad) / k


def main() -> None:
    args = parse_args()
    source = args.data_dir / "03_screening_evaluation" / "candidate_table_117.csv"
    data = pd.read_csv(source, dtype={"sample_id": str})

    top20 = data.nsmallest(20, "pred_rank")
    tau, p_value = kendalltau(top20["pred_rank"], top20["tcad_rank"])
    result = {
        "candidate_count": int(len(data)),
        "top20_kendall_tau": float(tau),
        "top20_kendall_p_value": float(p_value),
        "top8_screening_fraction_percent": 100.0 * 8.0 / len(data),
        "top_k_retention_percent": {str(k): topk_retention(data, k) for k in (8, 10, 15, 20)},
    }

    output_dir = Path("/results") if Path("/results").is_dir() else Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "screening_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
