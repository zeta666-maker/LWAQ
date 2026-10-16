# CKAA + AQCL — Continual Learning & Quantization-Aware Continual Learning

This repository contains two papers implemented and verified on a **tabular fault-diagnosis
dataset** (`fault_csv`, 9 fault classes → 5 incremental tasks `2/2/2/2/1`):

| Entry point | Paper | Core idea |
| --- | --- | --- |
| `train_eval.py` | **Algorithm 1 — CKAA** (TIP 2026, arXiv:2507.09471) | Cross-subspace Knowledge Alignment & Aggregation; task-ID-free inference via Task-Confidence-guided Mixture-of-Adapter (TC-MoA) |
| `train_aqcl.py` | **Algorithm 2 — AQCL** | Adaptive Quantization for Stable Knowledge Acquisition in Quantization-Aware Continual Learning (hand-written QAT + RPQ + SAOU) |

Both algorithms share the same CKAA backbone (ViT-B/16 with a 1D tabular tokenizer,
shared/task prompts, adapters, prototypes) and are kept **physically separated**: the CKAA
path in `train_eval.py` is AQCL-free, and AQCL only lives in `train_aqcl.py` + `utils/aqcl.py`.

---

## 1. Repository structure

```text
CKAA_TIP2026-main/
├── README.md
├── requirements.txt          # Python dependencies (verified versions)
├── .gitignore                # excludes logs/, _aqcl_work/, A_CLData/, weights, caches
├── .env.example              # environment-variable template
├── LICENSE
├── framework.png
├── train_eval.py             # Algorithm 1 (CKAA) entry point
├── train_aqcl.py             # Algorithm 2 (AQCL) entry point
├── scripts/                  # ready-to-run startup scripts
│   ├── prepare_data.sh / .ps1        # build fault_csv from raw CSVs
│   ├── run_ckaa.sh / .ps1            # run Algorithm 1
│   └── run_aqcl_matrix.sh / .ps1     # run Algorithm 2 (TC-MoA matrix)
├── utils/
│   ├── aqcl.py               # AQCL: uniform quantizer, QAT, RPQ, SAOU, deployment report
│   ├── aqcl_instrument.py    # per-layer quantization instrumentation
│   ├── vit_builder.py        # ViT, prompts, adapters, prototypes, losses, tabular tokenizer
│   ├── dataset_builder.py    # dataset routing (fault_csv / image datasets)
│   ├── tabular_data.py       # CSV -> windows converter + tabular Dataset
│   ├── continual_manager.py  # class-incremental task construction
│   ├── mod_adam.py           # projection-aware Adam
│   ├── logging.py            # console/file logger
│   ├── misc.py               # metrics helpers
│   └── osutils.py
├── clip/                     # CLIP model/tokenizer (used by the text-branch of the backbone)
├── tools/
│   ├── run_aqcl_tcmoa_matrix.py      # Algorithm 2 matrix runner (TC-MoA protocol built-in)
│   ├── prepare_tabular_dataset.py    # CSV -> fault_csv converter CLI
│   ├── run_validity.py               # shared validity guard (NaN / incomplete-task detection)
│   └── ...                           # analysis / inspection helpers
├── docs/
│   ├── AQCL.md                       # Algorithm 2 implementation record
│   ├── CKAA_AUDIT.md                 # Algorithm 1 audit
│   ├── FINAL_RESULTS_BOTH_PAPERS.md  # authoritative reproduced results
│   ├── FINAL_TWO_PAPER_RUN.md        # two-paper run record
│   └── archive/                      # historical handoff / checkpoint notes
├── docs_patent/              # patent source (Chinese)
├── docs_word/                # Word-report exports
└── A_CLData/                 # generated at runtime (git-ignored)
    └── fault_csv/            # processed windows/labels/meta.json
```

---

## 2. Environment

- **Python 3.11**, **PyTorch 2.11.0 + CUDA 12.8** (verified). Other recent PyTorch versions
  should work, but the exact versions below were used to produce the reported numbers.
- A CUDA-capable GPU is strongly recommended (the ViT backbone is large).

```bash
# 1) PyTorch with CUDA (adjust to your driver; see https://pytorch.org/get-started/locally/)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 2) Remaining dependencies
pip install -r requirements.txt
```

The pretrained ViT weights (`vit_base_patch16_224.augreg_in21k`) are downloaded automatically
from Hugging Face Hub by `timm` on first use, and cached under `~/.cache/huggingface`.

> **Windows PowerShell note:** if `.ps1` scripts are blocked by the execution policy
> (`running scripts is disabled on this system`), run them with
> `powershell -ExecutionPolicy Bypass -File scripts\run_ckaa.ps1` (same for the other scripts),
> or use the equivalent raw `python ...` command shown in each section below — those always work.

---

## 3. Data preparation

The dataset is built from raw vibration / pressure CSV streams. The raw data is **not
committed** (it is large and kept locally); you must provide it under `data_1_S1/` and
`data_2_S1/`:

```text
<RAW_ROOT>/
├── data_1_S1/     # one vibration signal column per CSV
└── data_2_S1/     # pressure X/Y/Z columns (columns 1:4)
```

By default `<RAW_ROOT>` is the **parent directory of this repository** (`..`). To use another
location, set the `DATA_RAW_ROOT` environment variable (see `.env.example`).

Build the processed `fault_csv` dataset (window length 1568, stride 1568, 4 channels
`[vibration, pressure_x, pressure_y, pressure_z]`, 80/20 train/eval split):

```bash
# Linux / macOS
bash scripts/prepare_data.sh
# Windows PowerShell
.\scripts\prepare_data.ps1
# or directly:
python tools/prepare_tabular_dataset.py --raw-root .. --output-root A_CLData/fault_csv --overwrite
```

This writes `A_CLData/fault_csv/{train,eval}_{windows,labels}.npy`, `*_index.csv`, and a
**portable** `meta.json` whose file paths are relative to the dataset directory (no
machine-specific absolute paths).

---

## 4. Algorithm 1 — CKAA

### 4.1 Background

CKAA (Cross-subspace Knowledge Alignment and Aggregation) is a class-incremental learning
framework on ViT. It builds task-specific prompts/adapters and aligns them across subspaces
(cross-subspace alignment) to reduce inter-task interference, and it updates shared prompts
in the null space of previously seen tasks to protect old knowledge. At inference it uses a
confidence-weighted **Mixture-of-Adapter (TC-MoA)** so no external task ID is required.

### 4.2 Core flow

1. Build 5 incremental tasks from the 9 fault classes (`2/2/2/2/1`).
2. For each task, train the task prompt/adapter/head while freezing the backbone and (after
   the first task) the shared prompt.
3. After each task, evaluate all seen tasks and record
   `class_inc_last_acc / class_inc_avg_acc / class_inc_last_forg`.
4. Select the task adapter with a trained task router (`--eval-trained-task-router true`).

### 4.3 Run (training + evaluation, one command)

```bash
# Linux / macOS
bash scripts/run_ckaa.sh
# Windows PowerShell
.\scripts\run_ckaa.ps1
# or directly:
python train_eval.py \
  -d fault_csv -t 5 -m vit_base_patch16_224.augreg_in21k \
  -b 32 -e 12 -jt 0 -je 0 -et 1 \
  --eval_batch_size 64 --temperature 28.0 \
  --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed \
  --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 \
  --eval-tool adapter \
  --eval-trained-task-router true --eval-local-task-head true --eval-task-weight false \
  --freeze-shared-prompts-after-first true --prototype-confidence-weight 0.0 \
  --seed 2024 --save-model false --logs-dir logs -sf ckaa_paper1_s2024
```

### 4.4 Key parameters

| Flag | Value | Meaning |
| --- | --- | --- |
| `-d` / `--dataset` | `fault_csv` | tabular fault dataset |
| `-t` / `--num_tasks` | `5` | 5 tasks, class split `2/2/2/2/1` |
| `-e` / `--epochs` | `12` | epochs per task |
| `-m` / `--model` | `vit_base_patch16_224.augreg_in21k` | ViT-B/16 backbone |
| `-b` | `32` | training batch size |
| `-jt` / `-je` | `0` | DataLoader workers = 0 (most portable) |
| `--temperature` | `28.0` | logit temperature |
| `--lr` / `--lr_scale` | `0.005` / `0.2` | base lr; scale for `patch_embed` |
| `--null_eta1/2` | `0.999` | null-space update decay |
| `-tc` / `--eval-tau` | `2.0` | TC-MoA temperature |
| `--eval-tool` | `adapter` | adapter-based evaluation |
| `--eval-trained-task-router` | `true` | trained task router for adapter selection |
| `--eval-local-task-head` | `true` | use the selected task head |
| `--eval-task-weight` | `false` | do not weight heads by router confidence |
| `--freeze-shared-prompts-after-first` | `true` | freeze shared prompt after task 1 |
| `--prototype-confidence-weight` | `0.0` | prototype confidence disabled |
| `--seed` | `2024` | random seed |

### 4.5 Evaluation metrics

- **Last-acc** (`class_inc_last_acc`): class-incremental accuracy after the final (5th) task.
- **Avg-acc** (`class_inc_avg_acc`): average class-incremental accuracy over the 5 tasks.
- **Forgetting** (`class_inc_last_forg`): backward-transfer forgetting of earlier tasks.

### 4.6 Reproduced result

| Metric | Value |
| --- | ---: |
| Last-acc | **95.99%** |
| Avg-acc | **97.92%** |
| Forgetting | **2.37%** |

Log: `logs/ckaa_paper1_s2024.txt` (produced by the command above).

---

## 5. Algorithm 2 — AQCL

### 5.1 Background

AQCL studies quantization-aware continual learning: it keeps a quantized model (asymmetric
uniform quantization + straight-through QAT) and adds three mechanisms so that low-bit
training does not destroy previously learned tasks:

- **RPQ** (sensitivity/robustness-aware bit allocation): allocate higher bit-width to
  quantization-sensitive layers.
- **SAOU** (stable anti-overfitting / orthogonal updates): project weight updates onto a
  subspace that avoids overwriting old tasks.
- **TC-MoA** evaluation protocol: confidence-weighted adapter mixture, deterministic across
  runs (no learned RandomForest router), so different bit-widths are directly comparable.

### 5.2 Core flow

1. Run the same CKAA backbone/adapters as Algorithm 1, but with quantization enabled
   (`train_aqcl.py`).
2. One warm-up epoch in FP32, then the target bit-width is enabled.
3. Per-task: Fisher/activation calibration → bit allocation (RPQ) → gradient modulation
   (SAOU) → QAT training.
4. After each task, print the `AQCL deployment:` line (model size / GFLOPs / GBOPS / bit map)
   and the same `class_inc_*` metrics.

### 5.3 Run (TC-MoA comparison matrix)

The runner `tools/run_aqcl_tcmoa_matrix.py` already embeds the TC-MoA protocol; run it
without modifying its internal commands:

```bash
# Linux / macOS
bash scripts/run_aqcl_matrix.sh
# Windows PowerShell
.\scripts\run_aqcl_matrix.ps1
# or directly:
python tools/run_aqcl_tcmoa_matrix.py \
  --cases fp32,fixed8,fixed4,fixed2 \
  --seeds 2024,2025 \
  --epochs 8
```

### 5.4 Key parameters (built into the runner)

| Parameter | Value | Meaning |
| --- | --- | --- |
| `--cases` | `fp32,fixed8,fixed4,fixed2` | no-quant / uniform 8 / 4 / 2-bit |
| `--seeds` | `2024,2025` | two seeds per case |
| `--epochs` | `8` | epochs per task |
| eval protocol | `--eval-trained-task-router false --eval-task-weight true` | TC-MoA confidence mixture |
| `--aqcl-fisher-batches` | `8` | Fisher/sensitivity sampling |
| `--aqcl-warmup-epochs` | `1` | FP32 warm-up before quantization |
| `--aqcl-calibrate-batches` | `4` | activation-range calibration |
| `--aqcl-per-token-activation` | `true` | per-token activation quantization |

### 5.5 Reproduced results (TC-MoA protocol)

| Case | seed | Last-acc | Avg-acc | Forgetting | size | bits |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fp32 | 2024 | 84.88 | 93.43 | 14.24 | — | — |
| fp32 | 2025 | 83.33 | 93.82 | 12.82 | — | — |
| fixed8 | 2024 | 80.87 | 90.98 | 16.93 | 88.89 MB | [8] |
| fixed8 | 2025 | 86.01 | 93.35 | 13.69 | 88.89 MB | [8] |
| fixed4 | 2024 | 68.85 | 83.94 | 31.25 | 45.54 MB | [4] |
| fixed4 | 2025 | 69.76 | 85.57 | 29.43 | 45.54 MB | [4] |
| **fixed2** | **2024** | **INVALID — diverged (`loss=nan` on task 3)** | | | | [2] |
| fixed2 | 2025 | 42.19 | 56.85 | 35.28 | 23.87 MB | [2] |

> **fixed2 seed 2024 NaN note (matches `docs/FINAL_RESULTS_BOTH_PAPERS.md`):** the 2-bit run
> with seed 2024 diverges — task 3 reports two `loss=nan` epochs (`qkv_grad_norm=nan`). The
> validity guard (`tools/run_validity.py`) therefore marks it **INVALID** and it is excluded
> from the summary. This is a known instability of the 2-bit operating point: some seeds abort
> or diverge, while the seeds that finish span a wide range. The **valid** 2-bit seed 2025
> result is **Last-acc 42.19 / Avg-acc 56.85 / Forgetting 35.28**.

Expected trend (consistent with the AQCL paper): **fp32 ≥ 8-bit ≥ 4-bit > 2-bit**, with model
size and GBOPS decreasing monotonically as bit-width drops (88.89 MB → 45.54 MB → 23.87 MB).

---

## 6. Configuration

Copy `.env.example` to `.env` (git-ignored) and override as needed:

| Variable | Default | Used by |
| --- | --- | --- |
| `DATA_RAW_ROOT` | `..` (repo parent) | `scripts/prepare_data.*` |
| `CUDA_VISIBLE_DEVICES` | `0` | all training |
| `SEED` / `EPOCHS` / `CKAA_SUFFIX` | `2024` / `12` / `ckaa_paper1_s<seed>` | `scripts/run_ckaa.*` |
| `AQCL_CASES` / `AQCL_SEEDS` / `AQCL_EPOCHS` / `AQCL_PREFIX` | `fp32,fixed8,fixed4,fixed2` / `2024,2025` / `8` / `tc` | `scripts/run_aqcl_matrix.*` |

All data/code paths are **relative** (`A_CLData/fault_csv`, `logs/`, `scripts/...`); no
machine-specific absolute paths are hardcoded in the source or the generated `meta.json`.

---

## 7. Citation

```bibtex
@article{he2025ckaa,
  title={Ckaa: Cross-subspace knowledge alignment and aggregation for robust continual learning},
  author={He, Lingfeng and Cheng, De and Ma, Zhiheng and Wang, Huaijie and Zhang, Dingwen and Wang, Nannan and Gao, Xinbo},
  journal={IEEE Transactions on Image Processing (TIP)},
  year={2026}
}
```
