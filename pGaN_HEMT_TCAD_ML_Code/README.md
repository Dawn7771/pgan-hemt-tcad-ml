# Post-Training Evaluation Utilities for p-GaN-Gate HEMT Surrogate Analysis

This compact public code release reproduces the principal post-training evaluation metrics reported in the associated manuscript:

- Kendall's tau and Top-k retention for the 117-candidate conditional screening experiment.
- Region-wise Vth classification accuracy and mean absolute error from saved independent-test predictions.
- Fixed-configuration CatBoost training and inference for the three-region Vth surrogate.

## Scope

The scripts operate on processed tables supplied in the accompanying dataset. TCAD deck generation, simulator setup, the complete curve-point expansion pipeline, hyperparameter searches, and internal design-selection implementation are outside the scope of this compact public release.

## Public-release scope

The package is designed to support verification of the main reported screening and threshold-diagnostic results, rather than to disclose the complete research-development environment. It therefore includes fixed-configuration CatBoost training and evaluation utilities, but excludes raw TCAD decks, simulator automation, internal tuning code, and other unpublished development materials.

## Requirements

Python 3.10 or later with the packages listed in `requirements.txt`.

## Run

Place this code folder beside `pGaN_HEMT_TCAD_ML_Dataset`, then run:

```bash
python evaluate_screening.py
python evaluate_vth_diagnostics.py
python train_catboost_vth.py
```

Alternatively, specify the data directory explicitly:

```bash
python evaluate_screening.py --data-dir /path/to/pGaN_HEMT_TCAD_ML_Dataset
python evaluate_vth_diagnostics.py --data-dir /path/to/pGaN_HEMT_TCAD_ML_Dataset
python train_catboost_vth.py --data-dir /path/to/pGaN_HEMT_TCAD_ML_Dataset
```

Each script writes its results to `results/` in the code directory. The public training script uses the fixed final configuration; it intentionally excludes hyperparameter-search utilities, TCAD automation, and output-curve-point modeling.

The generated `results/` directory is reproducible output and need not be uploaded as an additional input file.

## Code Ocean Capsule

For a Code Ocean Capsule, upload the contents of this directory directly to the `code` area, upload `pGaN_HEMT_TCAD_ML_Dataset` as a folder to the `data` area, and set `run.sh` as the File to Run. The master script invokes all three public reproductions and writes outputs to the Capsule `/results` area.
