#!/usr/bin/env bash
# Master script for Code Ocean reproducible execution.

set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/data/pGaN_HEMT_TCAD_ML_Dataset}"

python "${CODE_DIR}/evaluate_screening.py" --data-dir "${DATA_DIR}"
python "${CODE_DIR}/evaluate_vth_diagnostics.py" --data-dir "${DATA_DIR}"
python "${CODE_DIR}/train_catboost_vth.py" --data-dir "${DATA_DIR}"
