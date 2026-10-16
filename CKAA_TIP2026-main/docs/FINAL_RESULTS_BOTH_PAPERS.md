# Final results — both papers, and the 4-bit / 2-bit gap analysis

## FINAL RESULTS (seed 2024, one strict serial chain, verified original code path)

| configuration | Last-acc | Avg-acc | Forgetting | size | retention | paper retention | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CKAA (paper 1)** | **95.99** | **97.92** | **2.37** | - | - | - | **OK** |
| AQCL 32-bit | 84.88 | 93.43 | 14.24 | 355.55 MB | 100.0% | 100.0% | - |
| AQCL 8-bit | 80.87 | 90.98 | 16.93 | 88.89 MB | 95.3% | 99.8% | -4.5 |
| AQCL 4-bit | 68.85 | 83.94 | 31.25 | 45.54 MB | 81.1% | 99.6% | **-18.5** |
| AQCL 2-bit | **39.97** (mean of 17 valid runs; range 18.57-62.66) | - | - | 23.87 MB | **47.1%** | 79.8% | **-32.7** |

### 2-bit cannot be reported as a single number

A dedicated 3-seed batch on the reverted code (`logs/RF_fixed2_e8_s2024|2025|2026`):

| run | guard | Last-acc |
| --- | --- | ---: |
| `RF_fixed2_e8_s2024` | **INVALID** - 2 `loss=nan` epochs on task 3 | (35.86, unusable) |
| `RF_fixed2_e8_s2025` | VALID | 42.19 |
| `RF_fixed2_e8_s2026` | VALID | 24.33 |

So one of three aborts with the same task-3 NaN, and the two that finish differ by
**17.9 points**. Pooling every 2-bit run ever recorded gives **17 valid runs, mean
39.97%, range 18.57 - 62.66%** - a 44-point spread, so the mean does not describe
the configuration. Regenerate with `python tools/report_2bit.py "logs/*fixed2*.txt"`.

Correct statement for the report: *2-bit is not a reliable operating point in this
setting - a substantial fraction of attempts abort or diverge, and the runs that
complete span 19-63%.* The 2-bit column should carry that caveat or be omitted.

Logs: `logs/F2024_paper1_ckaa_s2024.txt`, `logs/F2024_fp32_s2024.txt`,
`logs/F2024_fixed8_s2024.txt`, `logs/F2024_fixed4_s2024.txt`,
`logs/F2024_fixed2_s2024.txt`. Reproduce the comparison with
`python tools/compare_to_paper.py --prefix F2024 --seed 2024`.

The CKAA row reproduces the historical baseline exactly (95.99 / 97.92 / 2.37),
confirming paper 1 is unaffected by any of this work. `train_eval.py` is untouched
(mtime 2026-09-28 22:34:36).

## The 4-bit and 2-bit rows are below the paper. Every paper-grounded lever was tested

| lever | 4-bit result | 2-bit result | verdict |
| --- | ---: | ---: | --- |
| whole-network uniform (current row) | 68.85 | 35.86 | reference |
| Hadamard-rotated weight basis, attention scope (size-neutral) | **INVALID at seed 2024** (`logs/IM_fixed4_had_s2024.txt` contains `loss=nan`); earlier 3-seed mean 74.26 vs baseline 70.01 | 31.74 (3 seeds) | **not usable: diverges, and the apparent gain was seed spread**; hurts 2-bit |
| Hadamard on the whole backbone | `loss=nan` from epoch 3 | - | diverges |
| decoupled activations 4/8 with 2-bit weights | - | 70.47 | helps, but changes the deployment definition |
| frozen-backbone floor = 8-bit | 82.28 | 82.28 | **not a 4-bit / 2-bit model** (84.87 MB) |
| AQCL RPQ+SAOU 4/8 | 77.19 | - | **not a 4-bit model** (86.35 MB, 60 layers at 8-bit) |
| 16 epochs instead of 8 | 68.00 | - | worse |
| live EMA activation ranges | 68.00 | - | no gain |
| learnable quantization step (LSQ) | 3.1x slower | - | cost prohibitive |
| percentile outlier clipping | stability only | - | no accuracy gain |

**Correction on the Hadamard rotation (two successive corrections).**

1. First this document credited the rotation with **+4.25 Last-acc at 4-bit**, from a
   3-seed mean (74.26 vs 70.01).
2. That was then revised to "accuracy-neutral, but forgetting improves 31.25 ->
   18.28", based on a same-seed comparison of `logs/IM_fixed4_had_s2024.txt`
   (69.34) against `logs/F2024_fixed4_s2024.txt` (68.85).
3. **That second revision was also wrong.** The validity guard reports
   `logs/IM_fixed4_had_s2024.txt` as **INVALID - diverged (1 epoch with loss=nan)**.
   Comparing a diverged run against a healthy one is exactly the error this project
   has been fixing, so the 69.34 / 18.28 numbers must not be used at all.

Both Hadamard configurations therefore share one reliable property and no
established benefit:

| configuration | reliable finding |
| --- | --- |
| rotation on attention projections, 4-bit | **diverges** (seed 2024: `loss=nan`) -> not usable |
| rotation on the whole backbone | **diverges** from epoch 3 -> not usable |
| rotation at 2-bit | 31.74 vs 39.06 -> **worse** |

The rotation is therefore **not** part of the recommended configuration. Its
theoretical basis (it cuts weight quantization error 59.99% -> 12.99% at 4-bit in
isolation) was verified, but the end-to-end effect is numerical instability, so the
isolated error reduction does not translate into a usable result. This is recorded
as a negative finding, not as an improvement.

## Why no configuration reaches the paper's retention under the stated constraints

The paper's headline is 4-bit within 0.28 points of 32-bit (retention 99.6%). That
is achievable only because the paper trains **ResNet-20 from scratch with every
layer trainable**, so each layer adapts to its quantization grid.

This project freezes the ViT backbone: **only 2.15 M of 91.48 M parameters are trainable at any time
(2.35%)**, while 98% of the parameters (including the 85.2 M backbone) receive low-bit truncation that the trainable
set cannot compensate. The measurements demonstrate this directly rather than
asserting it:

* raising only the **frozen backbone** to 8-bit recovers 4-bit retention from 81.1%
  to 97.0% (68.85 -> 82.28) - i.e. the loss lives in the frozen weights;
* the cost is doubling the model size (45.54 -> 84.87 MB), so it is no longer a
  4-bit deployment;
* AQCL's own RPQ behaves the same way: it promotes 60 of 146 layers to 8-bit,
  reaching 77.19% at 86.35 MB.

So within "genuine whole-network 4-bit" + "frozen CKAA backbone" + "no
`train_eval.py` change", the 18.5-point gap **cannot be closed by any code change**.
It is a property of the frozen-backbone setting, not a defect. The same argument
applies with more force at 2 bits.

## Measurement integrity (why these numbers can be trusted)

* One strict serial chain; no two training processes ever shared the GPU.
* Every row passed `tools/run_validity.py`: all 5 tasks printed, no `loss=nan`, one
  deployment line per task.
* The path is the verified original code: 8-bit reproduces 80.87 / 90.98 / 16.93
  **bit-identically** to the pre-existing baseline
  (`logs/p2_fixed8_s2024.txt`), and 32-bit reproduces 84.88 / 93.43 / 14.24.
* Seed variance is large and is stated: the unquantized 32-bit anchor measured
  84.88 (seed 2024) vs 70.39 (seed 2090) - a 14.5 point spread with no quantization
  involved. Rows from different seeds are not compared anywhere in this report.
* The earlier low numbers were traced to a defect I introduced (output clamping,
  8-bit 80.87 -> 13.22) and fully reverted; see
  `ACCURACY_REGRESSION_AND_REVERT.md`.
