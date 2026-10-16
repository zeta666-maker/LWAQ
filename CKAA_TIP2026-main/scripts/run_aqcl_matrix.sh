#!/usr/bin/env bash
# AQCL (paper 2): TC-MoA comparison matrix (fp32 / fixed8 / fixed4 / fixed2).
#   Usage:  bash scripts/run_aqcl_matrix.sh
#   Env:    AQCL_CASES   comma-separated cases   (default: fp32,fixed8,fixed4,fixed2)
#           AQCL_SEEDS   comma-separated seeds   (default: 2024,2025)
#           AQCL_EPOCHS  epochs per task         (default: 8)
#           AQCL_PREFIX  log filename prefix     (default: tc)
set -euo pipefail
cd "$(dirname "$0")/.."          # repository root

CASES="${AQCL_CASES:-fp32,fixed8,fixed4,fixed2}"
SEEDS="${AQCL_SEEDS:-2024,2025}"
EPOCHS="${AQCL_EPOCHS:-8}"
PREFIX="${AQCL_PREFIX:-tc}"

python tools/run_aqcl_tcmoa_matrix.py \
  --cases "${CASES}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --prefix "${PREFIX}"
