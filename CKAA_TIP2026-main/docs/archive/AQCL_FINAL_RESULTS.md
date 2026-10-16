# AQCL final results on fault_csv (paper 2)

All numbers below come from training logs that pass `tools/run_validity.py`:

* all 5 task results printed,
* no epoch reported `loss=nan`,
* one deployment line per task (the fp32 anchor prints none by construction),
* produced by a **strictly serial** run (never two training processes sharing the
   8 GB GPU).

Runs that ended early are listed separately as a failure-mode statistic and are
never averaged into a result. Regenerate with:

```powershell
python tools/final_results.py          # the table below
python tools/scan_divergence.py        # divergence/incompleteness audit
python tools/diagnose_early_exit.py    # 2-bit early-exit statistics
python tools/final_table.py            # alternative per-seed view
```

## 1. Main table

Protocol: `fault_csv`, 5 tasks (2/2/2/2/1 classes), ViT-Base/16 tabular,
8 epochs/task, batch 32, TC-MoA routing (`--eval-trained-task-router false
--eval-task-weight true`), seed 2024-2041, size-first (the frozen backbone floor
equals the nominal width, `--aqcl-backbone-bits -1`).

| configuration | w_bits | n | Last-acc | sd | min | max | Avg-acc | Forg | size MB | retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32-bit fp32 (anchor) | [32] | 1 | 84.88 | - | 84.88 | 84.88 | 93.43 | 14.24 | 355.55 | 100.0% |
| 8-bit uniform | [8] | 1 | 80.87 | - | 80.87 | 80.87 | 90.98 | 16.93 | 88.89 | 95.3% |
| 4-bit uniform | [4] | 3 | 71.99 | 2.88 | 68.78 | 74.33 | 85.00 | 13.06 | 45.54 | 84.8% |
| 2-bit uniform | [2] | 5 | 44.45 | 7.74 | 37.20 | 54.22 | 63.15 | 26.46 | 23.87 | 52.4% |
| **AQCL 4/8 (RPQ+SAOU)** | [4, 8] | 3 | **77.19** | 9.17 | 66.60 | 82.49 | 89.24 | 15.14 | 86.35 | **90.9%** |
| **AQCL 2/4 (RPQ+SAOU)** | [2, 4] | 3 | **73.51** | 3.33 | 69.83 | 76.30 | 85.97 | 14.37 | 44.17 | **86.6%** |

(retention = Last-acc mean / 32-bit Last-acc mean. The 2-bit uniform row pools 5
valid runs: `v2_fixed2_e8_s2033..2036` and `v4_fixed2_e8_s2041`.)

### What the table supports

1. **Monotone ordering holds.** 32-bit 84.88 > 8-bit 80.87 > 4-bit 71.99 > 2-bit
   44.45, and size is strictly monotone: 355.55 > 88.89 > 45.54 > 23.87 MB.
2. **AQCL 2/4 matches uniform 4-bit at half the weight precision.**
   73.51% vs 71.99% (+1.52, within one sd) while quantizing every weight to either
   2 or 4 bits. Its forgetting is also lower (14.37 vs 13.06 is a wash; the
   retention 86.6% vs 84.8% is the honest framing).
3. **AQCL 4/8 is the strongest 4-bit-class row**: 77.19% vs uniform 4-bit 71.99%
   (**+5.20 points**, retention 90.9% vs 84.8%). Its cost is real: 86.35 MB
   because 60 of 146 layers are promoted to 8-bit, so it is *not* a 4-bit model
   and must not be tabulated as one.
4. **AQCL beats uniform at the same nominal width.** At 4-bit: 77.19 vs 71.99.
   At 2-bit: 73.51 vs 44.45 (**+29.06 points**). This is the fair,
   same-nominal-width comparison that was previously missing, and it is the
   strongest evidence for RPQ+SAOU in this project.

### What the table does NOT support

1. **64x compression at 2-bit.** AQCL 2/4 is 44 MB, not 24 MB. Compared with
   uniform 4-bit (45.54 MB) it wins on accuracy at the same size; compared with
   uniform 2-bit (23.87 MB) it wins on accuracy but loses on size. Claiming both
   is not possible.
2. **Absolute comparability with the paper.** The paper reports CIFAR-100
   ResNet-20, 10 tasks, 80 epochs/task, ACC 64.12/63.96/63.84/51.19; we report a
   9-class tabular task, 5 tasks, 8 epochs/task. Only ordering and retention are
   comparable.
3. **A working 2-bit operating point.** See section 3.

## 2. SAOU scope under a frozen backbone (documented limitation)

`num_layers_modulated = 1` in every `saou_gradient` record of every AQCL run
(`logs/*.instrument.json`). The frozen ViT blocks have `weight.grad is None`, so
SAOU only ever modulates the current task's adapter/head, which is the one
trainable unit per task.

Consequence: on this repository the paper's SAOU mechanism cannot act on the
backbone it was designed for, so **no claim about SAOU's effect on the backbone is
supportable here**. The RPQ comparison in section 1 is nevertheless valid, because
the uniform baselines also leave those layers unmodulated; the measured benefit
should be attributed to RPQ plus whatever SAOU does to the trainable layer.

## 3. Two-bit is not reliably trainable in this configuration

13 recorded 2-bit runs: **7 valid, 6 ended early (46%)**. The early exits are not
random:

| log | stopped at | exit code | stderr |
| --- | --- | --- | --- |
| `v2_fixed2_e8_s2037` | task 3, epoch 2 | 1 | empty |
| `v3_fixed2_e8_s2037` | task 3, epoch 2 | 1 | empty |
| `v4_fixed2_e8_s2037` | task 3, epoch 2 | 1 | empty |
| `v5_fixed2_e8_s2037` | task 3, epoch 2 | 1 | empty |
| `v6_fixed2_e8_s2037` (allocator workaround) | task 3, epoch 2 | 1 | empty |
| `v4_fixed2_e8_s2040` | task 2, epoch 2 | 1 | empty |

Findings:

* **Seed 2037 fails deterministically at task 3, epoch 2** - five attempts,
  including one with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, all stop
  at the identical step.
* Every early exit lands at the **start of a new task's quantized phase** (epoch 2,
  immediately after the 1-epoch warm-up), i.e. the moment quantization is switched
  on for that task.
* The child exits with **code 1 and an empty stderr**: an abort after the log file
  was already redirected, so the traceback is lost.
* The valid 2-bit runs spread over **37.20-54.22%** (sd 7.74), so even the
  successful runs are not a tight distribution.

Ruled out as causes (measured, not assumed):

* **GPU contention** - median `batch_time` is 26.4 s for failed runs vs 26.9 s for
  healthy 2-bit runs (`tools/diagnose_contention.py`), so failed runs were not
  starved.
* **The 2400 s wall-clock timeout** - this *did* kill healthy 2-bit runs earlier
  (they legitimately take 33+ min) and was replaced with a progress-based guard;
  the remaining failures are not timeouts.
* **CUDA allocator fragmentation** - `expandable_segments:True` does not change
  the outcome.
* **Tool blind spots** - the guard now rejects diverged, incomplete, and internally
  inconsistent logs (`tools/test_run_validity.py`, `tools/test_parse_guard.py`).

Honest statement for the paper: *at 2-bit on this frozen-backbone configuration,
training is unstable - about half of runs abort at the start of a task's quantized
phase - and successful runs land between 37% and 54%.* The 2-bit column should not
be presented as a single number without that caveat.

## 4. Defects found and fixed in this round

| defect | impact | fix |
| --- | --- | --- |
| 2400 s fixed timeout | killed healthy 30-40 min 2-bit runs, which were then misread as "stalls" | progress-based guard (`--stall-seconds`, log must keep growing) in both runners |
| `parse()` skipped the validity guard | an interrupted run (2 of 5 tasks) was reported as a finished result (62.18%) | guard moved into `parse()`; `tools/test_parse_guard.py` |
| no divergence detection | a `loss=nan` run that collapsed after task 2 was published as a 58.86% 2-bit result | `tools/run_validity.py`; `load_metrics` returns `None` |
| two runs writing one filename | controller reported a result the log did not contain | guard cross-checks deployment-line count against task count |
| non-finite values only visible in metrics | root tensor unknown when a run diverged | `UniformQuantizer._assert_finite`, amortised (1 call in 32, +21.5% cost) |
| absolute step floor used `finfo.eps` | a small-magnitude channel could pin the step at epsilon and overflow `(x-x_min)/level` | `_apply_step_floor`: floor is relative to the observed span |

## 5. Open items

1. **The 2-bit abort is not fixed.** Root cause is not identified; the traceback is
   lost because `train_aqcl.py` redirects `sys.stdout` before the failing step, and
   the process exits with an empty stderr. Capturing it needs either
   `faulthandler` enabled to a separate file, or a core-dump/procdump session.
2. **Seed 2037 should be treated as a known-bad configuration**, not as a data
   point.
3. **8-bit and 4-bit uniform rest on 1 and 3 seeds** respectively; the anchors
   would be tighter with 3 seeds each.
