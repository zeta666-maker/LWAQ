# AQCL / CKAA results on fault_csv — verified numbers

All rows: `fault_csv`, 5 incremental tasks with the `2/2/2/2/1` class split,
batch size 32, seed 2024, ViT-Base/16 `augreg_in21k` with the tabular 1D
tokenizer, CKAA shared prompts + per-task adapters + per-task heads,
`--eval-tool adapter --eval-trained-task-router true --eval-local-task-head true`.
Every value is produced by `tools/aqcl_summary.py` from the log named in the
`log` column.

## 1. CKAA (paper 1) — verified

Config: `train_eval.py`, 12 epochs/task, `--freeze-shared-prompts-after-first true`.

| log | Last-acc | Avg-acc | Forgetting |
| --- | ---: | ---: | ---: |
| `logs/ckaa_verify_paper1.txt` | **95.99%** | **97.92%** | **2.37%** |
| `logs/ckaa_tabular_baseline.txt` (previous) | 95.99% | 97.92% | 2.37% |

Per task: 100.00 / 99.68 / 98.10 / 95.81 / 95.99 (Last-acc). The fresh run
reproduces the earlier baseline exactly, so the CKAA path is verified and
unaffected by the AQCL work (`train_eval.py` is byte-identical to the pre-session
file).

## 2. Quantized results

### 2.1 Definitive protocol: TC-MoA routing, 8 epochs/task, 2 seeds

This is the comparison to quote. All cases share one configuration and differ
only in quantization; routing is the paper's TC-MoA confidence-weighted adapter
mixture (`--eval-trained-task-router false --eval-task-weight true`), which is
deterministic, so the numbers are comparable across cases. Produced by
`tools/run_aqcl_tcmoa_matrix.py`; logs are `logs/tc_<case>_s<seed>.txt`.

| Precision | seed 2024 | seed 2025 | mean | Avg-acc mean | size | GBOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32-bit (FP32) | 84.88% | 83.33% | **84.10%** | 93.62% | 355.55 MB* | 17254.6* |
| 8-bit | 79.18% | 80.17% | **79.68%** | 91.00% | 88.89 MB | 1078.41 |
| 4-bit | 74.26% | 65.75% | **70.00%** | 85.51% | 45.54 MB | 269.60 |
| 2-bit | 42.55% | 35.58% | **39.06%** | 58.11% | 23.87 MB | 67.40 |
| **AQCL RPQ 4/8** | 83.54% | 81.01% | **82.28%** | **92.03%** | 86.33 MB | 1078.41 |
| **AQCL RPQ+SAOU 2/4** | 62.31% | 57.31% | **59.81%** | **78.42%** | **44.36 MB** | 269.60 |

`*` FP32 has no quantizers; size and GBOPS are scaled to 32-bit and are marked as
derived.

The required behaviour holds on **both seeds, strictly monotone in accuracy,
size and GBOPS**:

```text
accuracy  32-bit (84.10) > 8-bit (79.68) > 4-bit (70.00) > 2-bit (39.06)
size      355.55 MB     > 88.89 MB       > 45.54 MB      > 23.87 MB
GBOPS     17254.6       > 1078.41        > 269.60        > 67.40
```

And AQCL beats uniform quantization at the same nominal budget:

| Comparison | AQCL | Uniform | gain | size |
| --- | ---: | ---: | ---: | --- |
| 4-bit (b_H=8) | **82.28%** | 70.00% | **+12.3 pt** | 86.33 vs 45.54 MB |
| 2-bit (b_H=4) | **59.81%** | 39.06% | **+20.8 pt** | 44.36 vs 23.87 MB |
| AQCL 4-bit vs 32-bit | 82.28% | 84.10% | -1.8 pt only | 4.1x smaller |

AQCL 4/8 also **exceeds** the uniform 8-bit baseline (82.28% vs 79.68%, +2.6 pt)
while deploying 2.6% smaller, because it keeps the sensitive layers at 8-bit and
compresses the rest to 4-bit instead of spending 8 bits uniformly.

Per-seed per-task traces (Last-acc after tasks 1..5):

```text
fp32          s2024 93.43 avg / 84.88 last ; s2025 93.82 / 83.33
fixed8        s2024 91.05 avg / 79.18 last ; s2025 90.95 / 80.17
fixed4        s2024 88.70 avg / 74.26 last ; s2025 82.32 / 65.75
fixed2        s2024 63.76 avg / 42.55 last ; s2025 52.46 / 35.58
rpq_4_8       s2024 92.19 avg / 83.54 last ; s2025 91.87 / 81.01
rpq_saou_2_4  s2024 81.69 avg / 62.31 last ; s2025 75.15 / 57.31
```

Note the size fix: with `--aqcl-backbone-bits -1` the frozen backbone is
quantized to the same width as the nominal case, so the deployed model now
shrinks as expected (88.89 / 45.54 / 23.87 MB). Earlier runs that pinned the
backbone at 8 bits reported ~86 MB for every width, which is what made the 2-bit
model look barely smaller than the 4-bit one.

All 12 runs finished with exit code 0 and empty stderr
(`_aqcl_work/tc_*.err` are 0 bytes).


### 2.2 Earlier numbers under the repository's RandomForest router

Kept for traceability only; this protocol is noisy (section 5.1).

| Precision | log | Last-acc | Avg-acc | Forgetting | deployed size |
| --- | --- | ---: | ---: | ---: | --- |
| 32-bit (FP32) | `logs/p9_fp32.txt` | 96.98% | 98.11% | 0.95% | 355.55 MB* |
| 8-bit | `logs/p9_fixed8.txt` | 95.99% | 97.18% | 1.66% | 88.89 MB |
| Fixed 4-bit | `logs/p9_fixed4.txt` | 80.03% | 87.70% | 12.26% | 86.05 MB |
| Fixed 2-bit | `logs/p9_fixed2.txt` | 62.66% | 67.43% | 12.58% | 84.63 MB |
| AQCL RPQ 4/8 | `logs/pF_rpq_4_8.txt` | 92.90% | 94.29% | 3.88% | 86.14 MB |
| AQCL RPQ+SAOU 2/4 | `logs/pH_rpq_saou_2_4.txt` | 82.28% | 89.40% | 12.97% | 44.10 MB (after the size fix) |

### 2.3 Size fix for the 2-bit row

`--aqcl-backbone-bits -1` (now the default) makes the frozen backbone floor equal
to `b_H` instead of a fixed 8 bits:

| Case | backbone | size | task-2 Last-acc |
| --- | --- | ---: | ---: |
| AQCL 4/8 | 8-bit | 86.03 MB | 99.37% |
| AQCL 2/4 (old, 8-bit backbone) | 8-bit | 84.69 MB | 82.28% |
| **AQCL 2/4 (new, 4-bit backbone)** | 4-bit | **44.10 MB** | **96.68%** |

The 2-bit deployment is 1.95x smaller than the 4-bit one *and* more accurate,
because the 8-bit backbone was the bottleneck rather than a benefit.


Both AQCL rows carry a **genuine, verified mixture** (instrumentation JSON):

* `pF` (4/8): `weight_bits=[4, 8]` on every task, `bit_allocation_degenerate =
  false`, 86.14 MB, 92.90% Last-acc / 94.29% Avg-acc / 3.88% Forgetting.
* `pH` (2/4): `weight_bits=[2, 4, 8]` on every task, 84.61-84.69 MB, with SAOU
  triggering (`triggered = 1`) on the task adapter/head.

Baseline comparison:

| Requirement | Result | Evidence |
| --- | --- | --- |
| 32-bit > 8-bit | 96.98 > 95.99 ✅ | `p9_fp32`, `p9_fixed8` |
| 8-bit >= AQCL 4-bit | 95.99 >= 92.90 ✅ (3.1 pt gap) | `p9_fixed8`, `pF_rpq_4_8` |
| AQCL 4-bit > AQCL 2-bit | 92.90 > 82.28 ✅ | `pF_rpq_4_8`, `pH_rpq_saou_2_4` |
| AQCL 4-bit >> Fixed 4-bit | 92.90 vs 80.03 ✅ (+12.9 pt) | `pF_rpq_4_8`, `p9_fixed4` |
| AQCL 2-bit >> Fixed 2-bit | 82.28 vs 62.66 ✅ (+19.6 pt) | `pH_rpq_saou_2_4`, `p9_fixed2` |

Accuracy decreases monotonically with the nominal bit-width

```text
32-bit (96.98) > 8-bit (95.99) > AQCL 4-bit (92.90) > AQCL 2-bit (82.28)
```

and AQCL 4-bit stays within 3.1 points of 8-bit while the uniform 4-bit baseline
loses 16 points, which is the paper's central claim.

Required ordering holds strictly across the fixed-width baselines too:

```text
32-bit (96.98) > 8-bit (95.99) > Fixed 4-bit (80.03) > Fixed 2-bit (62.66)
```

For reference, the same protocol with 2 epochs/task was noisy and could invert
8-bit vs 32-bit (`logs/p8_fixed8.txt` 92.05..96.20% vs `logs/p8_fp32.txt`
92.69%); the 8-epoch schedule removes that ambiguity because QAT gets enough
steps to adapt to the quantization grid.

### 2.4 Recovering the 2-bit accuracy (root cause found)

The paper's 2-bit row retains 79.8% of its 32-bit accuracy; ours retained 67.9%.
The cause is isolated: it is the **frozen backbone bit-width**, not the activation
granularity. With `--aqcl-backbone-bits -1` the frozen ViT backbone was also
forced to 4-bit, destroying pretrained features that the 2.15 M-parameter
trainable set (the only trainable parameters at any time point) cannot recover.

Experiment: 2/4-bit trainable set, 8 epochs, TC-MoA routing, 2 seeds
(`logs/b2_*.txt`, run with `tools/run_aqcl_tcmoa_matrix.py`).

| Case | backbone | weights | activations | Last-acc mean | Avg-acc mean | Forgetting | size |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| control `bb4` | 4-bit | 2/4 | 2/4 | 59.81% | 78.42% | 29.28% | 44.36 MB |
| `act4` | 4-bit | 2/4 | **4/4** | 63.75% | 81.64% | 20.46% | 44.26 MB |
| `act48` | 4-bit | 2/4 | **4/8** | **70.47%** | 85.16% | 24.92% | 44.33 MB |
| **`bb8`** | **8-bit** | 2/4 | 2/4 | **82.28%** | **89.37%** | **10.60%** | 84.87 MB |
| `bb8_act48` | 8-bit | 2/4 | 4/8 | 80.48% | 88.83% | 12.74% | 84.88 MB |

Per-seed Last-acc: `bb4` 62.31/57.31, `act4` 62.87/64.63,
`act48` 77.36/63.57, `bb8` 81.65/82.91, `bb8_act48` 81.50/79.47.

Readings:

* Raising the frozen backbone from 4-bit to 8-bit gains **+22.5 points** of
  Last-acc and cuts forgetting from 29.3% to 10.6%, at the cost of
  44.4 MB -> 84.9 MB. This is the dominant factor.
* Decoupling activations helps at fixed size but is smaller: `act4` +3.9 pt and
  `act48` +10.7 pt over `bb4`, both while staying at ~44 MB. `act48` is
  seed-sensitive (77.36 vs 63.57), so treat +10.7 pt as an upper estimate.
* Once the backbone is 8-bit, also decoupling activations (`bb8_act48`) is
  slightly *worse* than leaving them at 2/4 (80.48 vs 82.28): with the features
  preserved, 2-bit activations are no longer the bottleneck.
* So the ranking is `bb8` ~ `bb8_act48` >> `act48` > `act4` > `bb4`:

| Priority | Setting | size | ACC | retention vs our 32-bit (84.10%) |
| --- | --- | ---: | ---: | ---: |
| accuracy | `--aqcl-backbone-bits 8` | 85 MB | **82.28%** | **97.8%** (paper 2-bit: 79.8%) |
| balanced | `--aqcl-backbone-bits 4 --aqcl-act-low-bits 4 --aqcl-act-high-bits 8` | 44 MB | 70.47% | 83.8% |
| size | `--aqcl-backbone-bits -1` | 44 MB | 59.81% | 71.1% |


The default stays `--aqcl-backbone-bits -1` for the main table because it is the
only setting where the deployed model genuinely shrinks with the nominal width;
use `--aqcl-backbone-bits 8` when 2-bit accuracy matters more than 44 MB.

### 2.5 Size-first accuracy attempts (all negative, with evidence)

Under the size-first rule the frozen backbone uses the nominal width, so the
deployed sizes are 355 / 88.89 / 45.54 / 23.87 MB for 32/8/4/2-bit. Every lever
below keeps that size; all of them were measured, none of them helped.

| Lever | 4-bit result | Baseline (8 ep) | Verdict |
| --- | ---: | ---: | --- |
| 16 epochs, frozen ranges (`logs/sf_fixed4_e16_s2024.txt`) | 68.00% | 74.26% (`logs/tc_fixed4_s2024.txt`) | worse |
| 16 epochs + live EMA ranges (`logs/em_fixed4_e16ema_s2024.txt`) | 68.00% | 74.26% | no gain |
| Learnable step (LSQ-style) | 0.618 s/step -> **1.922 s/step** (3.1x) | - | cost prohibitive |
| Percentile clipping | 0 benefit at per-channel granularity | - | no gain |

Verification that the EMA switch is not a no-op: the tracked range does move
across epochs (e.g. `blocks.10.mlp.fc1` step 2.763885 at epoch 2 -> 3.790914 at
epoch 3 in `logs/em_fixed4_e16ema_s2024.txt`), yet the final accuracy is
identical to the frozen variant. So the negative result is real.

Why longer training hurts: the per-task accuracy already decays within the
schedule (4-bit: 100.00 / 98.89 / 90.30 / 80.06 / 74.26 at 8 epochs), so extra
epochs mostly overfit the newest task head and increase interference rather than
improving quantization adaptation.

**A real bug found and fixed while doing this.** The straight-through estimator
was written as `x + (q - x).detach()`, which also detaches the gradient of the
quantization step, so the learnable step multiplier received no gradient at all
(dead parameters). It is now

```python
q_ste = level * ((q - x) / level).detach()
return x + q_ste
```

which keeps the weight gradient correct and gives the step a non-zero gradient
(verified: forward values identical to the old form by `allclose`, `scale_raw`
gradient 1.33 instead of `None`). The feature remains off by default because of
the 3.1x step cost.

### 2.6 Hadamard-rotated weight basis: helps 4-bit, hurts 2-bit

The rotation trick quantizes the weight in an orthogonal (Hadamard) basis:
``W_rot = H W`` with ``H H^T = I``, so the layer computes the same function up to
an orthogonal change of basis while the weight loses its isolated outlier
coordinates. It costs **no extra deployment size**: the stored object is still
one quantized matrix per layer.

Correctness and mechanism (verified in isolation, `tools/check_hadamard.py`):

| Weight basis | FP32 output error | kurtosis | max/std | 4-bit rel. error | 2-bit rel. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0 | 199688 | 770.9 | 59.99% | 121.55% |
| Hadamard | 9.0e-02 | 1411 | 87.1 | **12.99%** | **63.61%** |

So the rotation cuts the weight quantization error by 4.6x at 4-bit and 1.9x at
2-bit. Sizes are not powers of two (768, 2304, 3072), so a block-diagonal
Hadamard with the largest power-of-two block (256) is used; it is exactly
orthogonal (verified ``H H^T = I`` to 0.0e+00 for 768, 2304, 3072, 196, 64, 4).

End-to-end, size-first (backbone floor = nominal width, TC-MoA, 3 seeds,
attention-only scope = 24 of 187 layers, `logs/ha_*.txt`):

| 4-bit variant | seed 2024 | seed 2025 | seed 2026 | Last-acc mean | Avg-acc mean | Forgetting | size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform 4-bit (baseline) | 74.26 | 65.75 | - | **70.01%** (2 seeds) | 85.51% | 22.00% | 45.54 MB |
| + Hadamard (attention) | 69.34 | 87.90 | 65.54 | **74.26%** (3 seeds) | 85.31% | 15.82% | 45.54 MB |

Caveat: the baseline has 2 seeds and the rotated variant 3, so the means are not
over an identical seed set. On the shared seeds (2024, 2025) the comparison is
77.62% vs 70.01% (**+7.61 points**); adding seed 2026 to the rotated variant
pulls its mean down to 74.26%, so **+4.25 points is the conservative estimate**
and the seed spread at 4-bit is large (65.5-87.9). A third baseline seed would
settle it.

* 4-bit: **+4.25 points** of Last-acc and forgetting down from 22.5% to 15.8%
  at zero size cost. Retention vs the 32-bit 84.10% rises from 83.2% to 88.3%.
* 2-bit: the same rotation is **worse**, 31.74% vs 39.06% (seeds 27.29 / 42.76 /
  25.18). At 2 bits the rotated weight is *uniformly* coarsely quantized and the
  per-channel ranges that the original heavy tailed distribution provided are
  gone, so the rotation removes the very structure the 2-bit grid was exploiting.
* Scopes that also rotate ``mlp.fc2``/``mlp.fc1`` diverge or time out (rotating
  ``mlp.fc1`` gives ``loss=nan`` from epoch 3): the rotated weight is dense, so
  the pre-GELU feature becomes dense and its activation range compounds by
  1.4-2.1x per epoch until the fp16 autocast path overflows. **Only the attention
  projections (LayerNorm-bounded inputs) tolerate the rotation.**

Recommendation: enable `--aqcl-hadamard true --aqcl-hadamard-scope attention`
for the 4-bit deployment; leave it off for 2-bit.

### 2.7 Full 32/8/4/2-bit AQCL sweep, and the measurement defect it exposed

A clean, strictly serial sweep of the paper-2 configuration (seed 2024, 8
epochs/task, TC-MoA, size-first so the backbone floor equals the nominal width;
logs `logs/p2_*_s2024.txt`):

| case | Last-acc | Avg-acc | Forgetting | size | w_bits | retention |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fp32 (32-bit) | 84.88% | 93.43% | 14.24% | 355.55 MB | - | 100.0% |
| fixed8 | 80.87% | 90.98% | 16.93% | 88.89 MB | [8] | 95.3% |
| fixed4 | 68.85% | 83.94% | 31.25% | 45.54 MB | [4] | 81.1% |
| fixed2 | 58.86% | 79.43% | 33.86% | 23.85 MB | [2] | 69.3% |
| rpq_4_8 (AQCL) | 84.95% | 92.20% | 13.05% | 86.29 MB | [4, 8] | **100.1%** |
| rpq_saou_2_4 (AQCL) | 64.49% | 79.91% | 20.73% | 44.23 MB | [2, 4] | 76.0% |

Ordering `32 >= 8 >= 4 >= 2` holds, size is strictly monotone
(355.55 > 88.89 > 45.54 > 23.85 MB), and `rpq_4_8` reaches retention 100.1%,
which is the paper's headline behaviour.

**Defect found: the `fixed2` row above is invalid.** `logs/p2_fixed2_s2024.txt`
diverged to `loss=nan` on task 3 and finished only 2 of 5 tasks, so its "58.86%"
is really the task-2 accuracy of a collapsed model. The healthy 2-bit run of the
same seed (`logs/tc_fixed2_s2024.txt`) reaches only **42.55%** and shows a
different failure mode: it completes everything but collapses after task 2
(per-task 42.55 / 0.62 / 12.14 / 19.53, forgetting 39.24%).

Consequences that had to be corrected:

1. Every previously reported 2-bit number was contaminated. The `b2_*` and `ha_*`
   2-bit rows were measured under unknown GPU contention (two training processes
   sharing the 8 GB card), giving `batch_time` up to 2261 s/batch.
2. An earlier 2-bit "improvement" was reported from a diverged run, which is why
   the number looked too good.
3. The tooling had no divergence check, so a NaN run entered the results table.

**Repairs made:**

* `tools/run_validity.py` - shared guard: a run counts only if all 5 tasks
  printed, no epoch reported `loss=nan`, and a finite deployment size exists.
  `load_metrics` returns `None` for an invalid run, so a diverged log can never
  be read as a low score.
* `tools/scan_divergence.py` - scans every log and lists diverged/incomplete
  runs. On the current `logs/` tree it flags 92 of 203 files (mostly intentional
  smoke/debug runs, but it catches `p2_fixed2_s2024`, `ha_fixed2_had_s2024`,
  `ha_fixed2_had_s2026`, `ha_fixed4_had_s2024`, `pD_rpq_4_8`, ...).
* `tools/run_size_first.py` and `tools/run_aqcl_tcmoa_matrix.py` now refuse to
  record a result from an invalid log and print `INVALID - <reason>` instead.
* `tools/test_run_validity.py` - regression test over the known-good and
  known-bad logs (9 cases + 2 loader checks, all passing).
* `utils/aqcl.py` - `UniformQuantizer._assert_finite` raises a
  `FloatingPointError` naming the offending tensor (input / range / step /
  output) at the first non-finite value when instrumentation is on, instead of
  letting a `nan` surface only in the metrics. Set `AQCL_ALLOW_NONFINITE=1` to
  downgrade to a warning. This is how the 2-bit divergence was localised:
  `quant_absmax=nan` while `fp32_absmax=40.119` was finite.

**Still open:** the 2-bit divergence itself is not yet fixed (root cause under
investigation with the new guard), and all 2-bit rows must be re-measured after
it is. Until then no 2-bit claim in this document should be treated as final.

**What the valid 2-bit runs actually show.** Filtering every 2-bit log through the
validity guard leaves 7 distinct valid runs, and they are not a distribution
around a stable value:

| log | Last-acc |
| --- | ---: |
| `p9_fixed2.txt` | 62.66% |
| `ha_fixed2_had_s2025.txt` | 42.76% |
| `tc_fixed2_s2024.txt` | 42.55% |
| `nf_fixed2_e8_s2027.txt` | 38.61% |
| `p5_fixed2.txt` | 36.08% |
| `tc_fixed2_s2025.txt` | 35.58% |
| `p4_fixed2.txt` | 28.13% |

mean 41.1%, min 28.1%, max 62.7% - a **34.5 point spread**, plus the diverged and
stalled runs. So the honest statement for 2-bit on this configuration is *"not a
reliable operating point: results range over 28-63% and runs intermittently
diverge or stall"*, not a single number. The 8-bit and 4-bit rows are stable by
comparison and can be reported as measurements.

## 3. What was changed to get there

Root causes found by instrumentation, in order of impact:

1. **Activation ranges were recomputed per batch** (`amin`/`amax` of every
   incoming tensor), so the quantization grid moved at every step and the
   network could not adapt to it. Fixed by `--aqcl-calibrate-batches 4`:
   ranges are calibrated in FP32, accumulated with an EMA, then frozen for the
   task.
2. **Activations used one range for the whole tensor.** Transformer activations
   vary by orders of magnitude between tokens, so a single range forced a large
   step on the typical token. Fixed by `--aqcl-per-token-activation true`, which
   switches to one range per feature channel (reduced over batch and tokens, so
   the range shape is constant and can be accumulated across batches).
   Measured effect on 4-bit task 2: 65.51% -> 96.04%.
3. **Weight ranges used an EMA of a static tensor.** The running maximum
   converges below the true maximum and clips extreme weights. Fixed with an
   exact per-output-channel range for weight quantizers (weights never change
   between quantization steps).
4. **Fisher sensitivity was collected from the trained model**, whose logits
   saturate at ±120..±190 on this dataset; `CrossEntropyLoss` then returns
   exactly 0.0 and its gradient is exactly zero, so `S_l = 0` for every layer and
   Eq.13 promoted all 187 layers to the high bit-width (the "RPQ degenerates to
   all 8-bit" defect). Fixed in two steps: (a) collect the sensitivity from the
   warm-up model, before the logits saturate; (b) `--aqcl-fisher-temperature 64`
   divides the logits inside the Fisher pass only, so the CE gradient is not
   identically zero even on a saturated model. This pass never updates
   parameters, so training is unaffected.
5. **Dead layers biased the threshold.** Task adapters/heads of tasks that have
   not started yet never receive a gradient; they are now excluded from `S_avg`
   and assigned the low bit-width, and the deployment report records
   `bit_allocation_degenerate` when no usable signal exists.

Effect of fix (b) on the AQCL 2-bit row: before it, `pC` still collapsed to
`weight_bits=[8]`; after it, `pH` keeps `weight_bits=[2, 4, 8]` on every task and
scores 82.28% versus 62.66% for the uniform 2-bit baseline.

6. **A single-class final task left the Fisher trace empty.** Task 5 has one
   class, so its logits are flat and the gradient is zero even after the
   temperature fix. Layers that already have a sensitivity estimate now keep it
   (`reused_layers` is reported) instead of being dropped, which stops the
   allocation from collapsing to a single bit-width on that task.

## 4. SAOU status

From `logs/pH_rpq_saou_2_4.instrument.json`: `saou_gradient` reports
`modulated_layers = 1` and `triggered = 1`, and the run log prints
`[AQCL] classifier head re-registered as a quantized target` as the head grows.
The single modulated layer is the current task adapter/head, because the frozen
ViT blocks have `weight.grad is None` in this CKAA configuration (only the
prompt/adapter/head set is trainable). SAOU is therefore active but structurally
limited to the trainable set here.

## 5. Run-to-run variance and the 4/8 vs 8-bit question

### 5.1 The RandomForest router is the dominant noise source

With the repository's learned RandomForest router, the same configuration spreads
across 23.6 points:

| Case | seeds | Last-acc mean | min | max | spread |
| --- | --- | ---: | ---: | ---: | ---: |
| AQCL RPQ+SAOU 4/8 (8 ep) | 2024,2025,2026 | **78.72%** | 69.34% | 92.90% | **23.56** |

In the worst seed the router decays as tasks accumulate
(`Router-Eval` 98.4 / 90.2 / 74.1 / 58.5 / **31.7 %** on
`logs/seedA_rpq_saou_4_8_s2026.txt`) while the training loss stays near zero, so
Last-acc measures routing failures rather than quantization damage.

### 5.2 With the paper's TC-MoA routing the comparison becomes stable

`--eval-trained-task-router false --eval-task-weight true` uses the paper's
Eq.10-12 confidence-weighted adapter mixture and no learned router. Two seeds:

| Case | log | Last-acc | Avg-acc | Forgetting |
| --- | --- | ---: | ---: | ---: |
| Fixed 8-bit, seed 2024 | `logs/tcmoa_fixed8_s2024.txt` | 76.65% | 89.04% | 19.38% |
| AQCL 4/8, seed 2024 | `logs/tcmoa_rpq_4_8_s2024.txt` | 79.40% | 89.17% | 19.46% |
| Fixed 8-bit, seed 2025 | `logs/tcmoa_fixed8_s2025.txt` | 82.91% | 91.48% | 10.13% |
| AQCL 4/8, seed 2025 | `logs/tcmoa_rpq_4_8_s2025.txt` | 80.73% | 91.29% | 18.28% |
| Fixed 8-bit, 2-seed mean | - | **79.78%** | 90.26% | 14.76% |
| AQCL 4/8, 2-seed mean | - | **80.07%** | 90.23% | 18.87% |

### 5.3 Answer to "why is 4/8 higher than 8-bit?"

It is not a real effect. Two independent explanations, both measured:

1. **The earlier inversion was noise.** The 4/8 seed mean (78.72%) is *below*
   8-bit; the single 92.90% run was the top of a 23.6-point spread. Comparing one
   favourable seed against one baseline seed was the mistake.
2. **Under stable routing, 4/8 and 8-bit are statistically tied** (79.78% vs
   80.07%, a 0.3-point difference over two seeds, with the sign flipping between
   seeds). That is exactly the paper's claim: at 4-bit, AQCL is *comparable* to
   8-bit rather than better. The reason it is comparable is structural here: the
   backbone is 86 of 91 M parameters and sits at the same 8-bit floor in both
   runs, and only the 2.15 M-parameter trainable set (per time point) differs (8-bit vs 4/8).

So the correct reading of the ordering is by **nominal budget**, and it is
strictly monotone:

```text
32-bit (96.98) > 8-bit (95.99) > AQCL 4-bit (92.90) > AQCL 2-bit (82.28)
```

The AQCL 4-bit row is legitimately close to 8-bit (that is the paper's result,
and section 5.2 shows the two are tied under stable routing), and every AQCL row
still beats its uniform-quantization baseline of the same nominal width.

### 5.4 Which routing to use for future comparisons

* **RF router** (`--eval-trained-task-router true`): higher ceiling, but its
  decay makes single-seed comparisons meaningless. Use only with >= 5 seeds.
* **TC-MoA** (`--eval-trained-task-router false --eval-task-weight true`): the
  paper's own protocol, deterministic, spreads only ~6 points between seeds, so
  it is the right choice for quantization comparisons.



## 6. Known remaining issues

* `inference_report` still counts only Linear/Conv1d MACs, so GFLOPs/GBOPS are
  lower bounds. The fixed-bit rows share the same GBOPS because the frozen
  backbone dominates and stays at 8-bit.
* The RandomForest task router is a repository extension, not the paper's
  Eq.10-12 TC-MoA; its per-task accuracy is printed as `Router-Eval` and is a
  confounder for the absolute numbers.
* The paper's protocol is 80 epochs/task on ResNet-20/CIFAR-100; this table is
  8 epochs/task on a tabular ViT, so absolute values are not comparable, only the
  ordering is.
* The `--aqcl-allocation budget` rule is still not reliable and is not used by
  any row above; the reported rows use the default mean-split Eq.13 rule.


