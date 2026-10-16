# FINAL run of both papers + issue list

## 1. Run 1 — CKAA (paper 1), `train_eval.py`

Command (pure CKAA entry, 12 epochs/task, 5 tasks, seed 2024):

```powershell
python -u .\train_eval.py -d fault_csv -t 5 -m vit_base_patch16_224.augreg_in21k `
  -b 32 -e 12 -jt 0 -je 0 -et 1 --eval_batch_size 64 --temperature 28.0 `
  --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed `
  --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 --eval-tool adapter `
  --eval-trained-task-router true --eval-local-task-head true `
  --eval-task-weight false --freeze-shared-prompts-after-first true `
  --prototype-confidence-weight 0.0 --seed 2024 --save-model false `
  --logs-dir logs -sf FINAL_paper1_ckaa_12e
```

Log: `logs/FINAL_paper1_ckaa_12e.txt`, exit code 0.

| Task | Last-acc | Avg-acc | Forgetting |
| --- | ---: | ---: | ---: |
| 1 | 100.00% | 100.00% | 0.00% |
| 2 | 99.68% | 99.84% | 0.63% |
| 3 | 98.10% | 99.26% | 0.16% |
| 4 | 95.81% | 98.40% | 2.74% |
| 5 | **95.99%** | **97.92%** | **2.37%** |

Identical to `logs/ckaa_tabular_baseline.txt` and `logs/ckaa_verify_paper1.txt`:
the CKAA path is deterministic and reproducible. Meets the target
(Avg-acc >= 90%, Forgetting < 10%). `train_eval.py` is unchanged
(mtime 2026-09-28 22:34:36).

## 2. Run 2 — AQCL (paper 2), `train_aqcl.py`, 2-bit accuracy-first

Command (RCAA/AQCL config identical to paper 1 except quantization; TC-MoA
routing, 8 epochs/task):

```powershell
python -u .\train_aqcl.py -d fault_csv -t 5 -m vit_base_patch16_224.augreg_in21k `
  -b 32 -e 8 -jt 0 -je 0 -et 1 --eval_batch_size 64 --temperature 28.0 `
  --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed `
  --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 --eval-tool adapter `
  --eval-trained-task-router false --eval-task-weight true `
  --eval-local-task-head true --prototype-confidence-weight 0.0 `
  --seed 2024 --save-model false --logs-dir logs `
  --aqcl-mode rpq_saou --aqcl-low-bits 2 --aqcl-high-bits 4 `
  --aqcl-backbone-bits 8 --aqcl-fisher-batches 8 --aqcl-warmup-epochs 1 `
  --aqcl-calibrate-batches 4 --aqcl-per-token-activation true `
  --aqcl-instrument true --aqcl-instrument-verbose false `
  -sf FINAL_paper2_aqcl_2bit
```

Log: `logs/FINAL_paper2_aqcl_2bit.txt`, exit code 0.

| Task | Last-acc | Avg-acc | Forgetting | deployed | w_bits |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 98.10% | 98.10% | 0.00% | 84.61 MB | [2, 4, 8] |
| 2 | 96.20% | 97.15% | 4.43% | 84.78 MB | [2, 4, 8] |
| 3 | 92.19% | 95.50% | 6.65% | 84.80 MB | [2, 4, 8] |
| 4 | 68.59% | 88.77% | 11.92% | 84.82 MB | [2, 4, 8] |
| 5 | **81.65%** | **87.35%** | **5.30%** | 84.82 MB | [2, 4, 8] |

## 3. Issue list

### P0 — methodological, must fix before any paper claim

1. **RPQ's benefit is not demonstrated against a fair baseline.** The only
   "AQCL beats uniform" comparison uses a *larger* nominal width
   (`rpq_4_8` 82.28% vs uniform 8-bit 79.68%): the small gain is not explained by
   RPQ. A four-way comparison at the same nominal width and same backbone floor
   (uniform 2 vs RPQ 2, uniform 4 vs RPQ 4) has never been run under the stable
   TC-MoA protocol. Evidence: `logs/tc_*.txt` vs `logs/b2_*.txt`.
2. **SAOU modulates one layer only.** Every `saou_gradient` record reports
   `modulated_layers = 1` (the current task adapter/head), because the frozen ViT
   blocks have `weight.grad is None`. SAOU's intended effect on the backbone is
   structurally unreachable here, so no SAOU claim can be made.
   Evidence: `logs/FINAL_paper2_aqcl_2bit.instrument.json`.
3. **Fisher sensitivity collapses on the last task.** In this run
   `fisher_warmup` reports `nonzero_layers = 0, reused_layers = 141` on task 5
   (single-class task, flat logits, zero gradient even with the temperature fix),
   so the task-5 allocation is a replay of task-4 estimates rather than a
   measurement.
4. **Evaluation routing is still the largest confounder.** With the repository's
   RandomForest router the same configuration spreads 23.6 points across seeds;
   with TC-MoA it is ~6 points. Absolute numbers therefore depend on a routing
   choice that is a repository extension, not the paper's Eq.10-12.
   Evidence: `logs/seedA_*.txt`, `logs/tc_*.txt`.
5. **No multi-seed statistics for most rows.** Only `rpq_saou_2_4` and the
   `b2_*` family have 2 seeds; the rest are single-seed, and section 5.1 shows a
   single seed is not evidence.

### P1 — results still worse than the paper, cause identified

6. **4-bit retention is far below the paper.** Ours 70.00/84.10 = 83% vs the
   paper's 63.84/64.12 = 99.6%. The paper quantizes a from-scratch ResNet-20
   where every layer can adapt; we quantize a frozen ViT backbone.
7. **2-bit retention below the paper on the size-first definition.**
   `--aqcl-backbone-bits -1`: 59.81/84.10 = 71% vs the paper's 79.8%. Only the
   accuracy-first definition (`8`) reaches 97.8%.
8. **BWT/forgetting universally worse.** Ours -12 to -30 vs the paper's -5 to -6.
   Forgetting is driven by head/router aggregation, not by quantization.
9. **Model sizes are not comparable to the paper.** Our "4-bit" deployment is
   86 MB vs the paper's 0.193 MB (450x) because the frozen backbone dominates and
   stays at 8-bit. We report model size honestly but it does not demonstrate
   edge-deployment compression.
10. **The 2-bit configuration is seed-sensitive.** `act48` scored 77.36 vs 63.57
    (13.8-point spread), `bb8` 81.65 vs 82.91. Conclusions at 2-bit need more
    seeds.

### P2 — reporting and efficiency gaps

11. **GBOPS is not informative.** `inference_report` counts Linear/Conv1d MACs
    only; attention `QK^T` and `attn @ V` are missing, and because the frozen
    backbone dominates and sits at 8-bit, GBOPS is identical (1078.41) for
    uniform 4, uniform 2 and RPQ 4/8 rows. See `AQCL.md` section 5.
12. **`Avg-acc` is not the paper's exact ACC.** The paper's ACC averages the
    per-task accuracies *after the final task*; our `Avg-acc` averages the
    historical Last-acc list. They are close but not the same formula; section 2.4
    computes the paper formula separately (`tools/collect_logs.py`
    `per_task` output).
13. **Claim hygiene.** Only the "accuracy/ordering" trend is supported. Any
    statement of the form "AQCL improves over uniform by X points" must be
    qualified with the nominal-width mismatch (issue 1).

### P3 — reproducibility / tooling

14. **PowerShell script execution is disabled** on this machine, so
    `tools/run_aqcl_comparison.ps1` cannot run; use
    `tools/run_aqcl_matrix.py` / `tools/run_aqcl_tcmoa_matrix.py`.
15. **The matrix runner must redirect child stdout to files.** Letting the child
    inherit the pipe deadlocks once the pipe buffer fills (observed as a false
    "hang" at task 5); the runner now redirects, keep that behaviour.
16. **Log flushing.** `utils/logging.py` writes through a buffered file handle,
    so a running job's log can appear frozen; `python -u` fixes stdout but not the
    log file. Do not diagnose a hang from the log file alone.
