#!/usr/bin/env bash
# CKAA (paper 1): train + evaluate on fault_csv, 5 incremental tasks (2/2/2/2/1).
#   Usage:  bash scripts/run_ckaa.sh
#   Env:    SEED         random seed           (default: 2024)
#           EPOCHS       epochs per task       (default: 12)
#           CKAA_SUFFIX  log filename suffix   (default: ckaa_paper1_s<seed>)
set -euo pipefail
cd "$(dirname "$0")/.."          # repository root

SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-12}"
SUFFIX="${CKAA_SUFFIX:-ckaa_paper1_s${SEED}}"

python train_eval.py \
  -d fault_csv -t 5 \
  -m vit_base_patch16_224.augreg_in21k \
  -b 32 -e "${EPOCHS}" -jt 0 -je 0 -et 1 \
  --eval_batch_size 64 \
  --temperature 28.0 \
  --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed \
  --null_eta1 0.999 --null_eta2 0.999 \
  -tc 2.0 \
  --eval-tool adapter \
  --eval-trained-task-router true \
  --eval-local-task-head true \
  --eval-task-weight false \
  --freeze-shared-prompts-after-first true \
  --prototype-confidence-weight 0.0 \
  --seed "${SEED}" --save-model false --logs-dir logs -sf "${SUFFIX}"
