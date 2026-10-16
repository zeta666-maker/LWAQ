#!/usr/bin/env bash
# Build the processed fault_csv dataset from the raw CSV streams.
#   Usage:  bash scripts/prepare_data.sh
#   Env:    DATA_RAW_ROOT  directory containing data_1_S1/ and data_2_S1/
#                          (default: the parent directory of this repository)
set -euo pipefail
cd "$(dirname "$0")/.."          # repository root

RAW_ROOT="${DATA_RAW_ROOT:-..}"
OUTPUT_ROOT="${OUTPUT_ROOT:-A_CLData/fault_csv}"

echo "Building fault_csv from: ${RAW_ROOT}"
python tools/prepare_tabular_dataset.py \
  --raw-root "${RAW_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --overwrite
