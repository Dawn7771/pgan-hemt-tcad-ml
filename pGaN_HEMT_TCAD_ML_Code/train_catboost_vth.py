"""Train the fixed CatBoost three-region Vth surrogate and evaluate it on saved test IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


FEATURES = [
    "t_AlGaN_nm", "x_AlGaN", "N_AlGaN_cm3",
    "t_pGaN_nm", "N_pGaN_cm3", "Lgd_um",
]
REGIONS = ["low", "mid", "high"]
SEED = 42
TAU = 0.98
ALPHA = 0.85


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


def sharpen(probabilities: np.ndarray) -> np.ndarray:
    scaled = np.power(np.clip(probabilities, 1e-12, 1.0), 1.0 / TAU)
    return scaled / scaled.sum(axis=1, keepdims=True)


def ordered_probabilities(model: CatBoostClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    return pd.DataFrame(raw, columns=model.classes_).reindex(columns=REGIONS).to_numpy(float)


def stage_predictions(train: pd.DataFrame, x: pd.DataFrame) -> np.ndarray:
    parameters = {
        "low": dict(depth=6, learning_rate=0.03, iterations=1200, l2_leaf_reg=3.0),
        "mid": dict(depth=5, learning_rate=0.05, iterations=900, l2_leaf_reg=2.0),
        "high": dict(depth=3, learning_rate=0.05, iterations=700, l2_leaf_reg=2.0),
    }
    predictions = np.zeros((len(x), len(REGIONS)))
    for column, name in enumerate(REGIONS):
        model = CatBoostRegressor(
            loss_function="RMSE", random_seed=SEED, verbose=False, allow_writing_files=False,
            **parameters[name]
        )
        subset = train.loc[train["true_region"].eq(name)]
        model.fit(subset[FEATURES], subset["vth_v"])
        predictions[:, column] = model.predict(x[FEATURES])
    return predictions


def main() -> None:
    args = parse_args()
    foms = pd.read_csv(args.data_dir / "02_tcad_outputs" / "foms.csv", dtype={"sample_id": str})
    samples = pd.read_csv(args.data_dir / "01_design_samples" / "samples_augmented_1600.csv", dtype={"sample_id": str})
    status = pd.read_csv(args.data_dir / "01_design_samples" / "final_sample_status.csv", dtype={"sample_id": str})
    test_predictions = pd.read_csv(
        args.data_dir / "04_reproducibility_metadata" / "vth_catboost_independent_test_predictions.csv",
        dtype={"sample_id": str},
    )
    training_membership = pd.read_csv(
        args.data_dir / "04_reproducibility_metadata" / "vth_fixed_training_membership.csv",
        dtype={"sample_id": str},
    )
    for frame in (foms, samples, status, test_predictions, training_membership):
        frame["sample_id"] = frame["sample_id"].str.zfill(6)

    valid_ids = set(status.loc[status["status"].eq("valid"), "sample_id"])
    data = foms.merge(samples, on="sample_id", how="inner")
    data = data.loc[data["sample_id"].isin(valid_ids) & data["vth_flag"].eq("bracketed")].copy()
    data = data.dropna(subset=[*FEATURES, "vth_v"])
    data["true_region"] = data["vth_v"].map(region)

    test_ids = set(test_predictions["sample_id"])
    train_ids = set(training_membership["sample_id"])
    train = data.loc[data["sample_id"].isin(train_ids)].copy()
    test = data.loc[data["sample_id"].isin(test_ids)].copy()
    if train_ids & test_ids or len(train) != 1002 or len(test) != 251:
        raise RuntimeError(f"Expected a 1002/251 split, received {len(train)}/{len(test)}.")

    classifier = CatBoostClassifier(
        loss_function="MultiClass", depth=5, learning_rate=0.04, iterations=350,
        l2_leaf_reg=5.0, random_seed=SEED, verbose=False, allow_writing_files=False,
    )
    classifier.fit(train[FEATURES], train["true_region"])
    probability = ordered_probabilities(classifier, test[FEATURES])
    predicted_region = classifier.predict(test[FEATURES]).reshape(-1)
    staged = stage_predictions(train, test)
    hard_index = np.array([REGIONS.index(label) for label in predicted_region])
    hard_prediction = staged[np.arange(len(test)), hard_index]
    soft_prediction = np.sum(staged * sharpen(probability), axis=1)
    prediction = ALPHA * soft_prediction + (1.0 - ALPHA) * hard_prediction

    result = test[["sample_id", "vth_v", "true_region"]].copy()
    result["pred_vth"] = prediction
    result["pred_region"] = predicted_region
    metrics = {
        "training_samples": int(len(train)),
        "test_samples": int(len(test)),
        "classification_accuracy_percent": 100.0 * float(accuracy_score(test["true_region"], predicted_region)),
        "r2": float(r2_score(test["vth_v"], prediction)),
        "mae_v": float(mean_absolute_error(test["vth_v"], prediction)),
        "tau": TAU,
        "alpha": ALPHA,
    }

    output_dir = Path("/results") if Path("/results").is_dir() else Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    result.to_csv(output_dir / "catboost_vth_test_predictions.csv", index=False)
    (output_dir / "catboost_vth_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
