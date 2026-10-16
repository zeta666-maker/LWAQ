# Why the accuracy dropped, and what was done about it

## Short answer

The drop was **caused by a "fix" I added myself**, not by the paper code. It is now
reverted, and the original numbers reproduce **exactly**.

## The regression, measured

Same configuration (8-bit, seed 2024, TC-MoA, 8 epochs/task). Only the quantizer's
numerical path differs:

| code path | Last-acc | Avg-acc | Forgetting |
| --- | ---: | ---: | ---: |
| **original code** (`logs/p2_fixed8_s2024.txt`) | **80.87** | **90.98** | **16.93** |
| + output clamping ON - my introduced defect | **13.22** | 32.91 | 46.44 |
| clamping OFF, other changes on | 78.69 | 89.26 | 20.41 |
| **`AQCL_LEGACY=1`, full original path** | **80.87** | **90.98** | **16.93** |
| **new default (legacy), no env var** (`logs/nv_fixed8_s2024.txt`) | **80.87** | **90.98** | **16.93** |

The mechanism: the clamp saturates every value outside the **calibrated** range onto
the range endpoints. The calibrated range is consistently narrower than the values
seen during a task's training, so most of the tensor is destroyed. The theoretical
justification ("real integer quantization saturates") does not survive measurement
here.

## What was changed

1. `clamp_output` now defaults to **OFF** (`AQCL_CLAMP_OUTPUT=1` re-enables it for
   experiments only).
2. `AQCL_LEGACY` now defaults to **ON**, restoring the complete original quantizer
   path: no span-based step floor, no observation sanitisation, no degenerate-range
   recovery. `AQCL_LEGACY=0` enables those later changes for A/B comparison.
3. The classifier head keeps the original per-tensor activation granularity
   (per-channel on the head was also measured to hurt).
4. `--aqcl-clip-percentile` defaults to 0. The per-channel quantile is a sort
   costing 171.6 ms per call versus 0.20 ms for min/max, which alone made a 2-bit
   epoch take 108 s instead of 12 s.

Everything kept is **diagnostic only** and cannot change numerics: the crash
traceback capture (`<log>.crash.txt`), the validity guard that rejects diverged or
incomplete runs, the exit-code capture, and the progress-based run guard that
replaced the wall-clock timeout.

## Verification that the revert is exact

```
original code      : 80.87 / 90.98 / 16.93   (logs/p2_fixed8_s2024.txt)
AQCL_LEGACY=1      : 80.87 / 90.98 / 16.93   (logs/lg_fixed8_s2024.txt)
new default        : 80.87 / 90.98 / 16.93   (logs/nv_fixed8_s2024.txt)
32-bit, re-run     : 84.88 / 93.43 / 14.24   (logs/chk_fp32_s2024.txt == p2_fp32_s2024.txt)
```

## A separate, real source of variation: seeds

The unquantized 32-bit anchor itself varies a lot:

| seed | log | Last-acc |
| --- | --- | ---: |
| 2024 | `logs/chk_fp32_s2024.txt` | 84.88 |
| 2090 | `logs/fin_fp32_s2090.txt` | 70.39 |

A **14.5 point** spread with no quantization involved. Any single-seed row must be
quoted with its seed, and rows from different seeds must not be compared.

## Final measurement in progress

`tools/run_final_sweep.py --seed 2024` runs, strictly serially on the verified path:

1. AQCL 32-bit -> 8-bit -> 4-bit -> 2-bit
2. CKAA paper-1 baseline (12 epochs) via the unmodified `train_eval.py`

`train_eval.py` remains untouched (mtime 2026-09-28 22:34:36).
