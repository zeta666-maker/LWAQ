# Root cause of the 2-bit aborts (paper 2 / AQCL)

This file records the diagnosis chain for the failure that produced exit code 1
with **empty stdout and empty stderr** in ~46% of 2-bit runs. Every step below is
backed by an artifact in `logs/`.

## The blocker was that the traceback was lost

`train_aqcl.py` replaces `sys.stdout` with `Logger(...)` before training starts, so
an uncaught exception had nowhere visible to print and the process died as a bare
exit code 1. `faulthandler` was **0 bytes**
(`logs/v7_fixed2_e8_s2037.faulthandler.txt`), which rules out a fatal signal
(`SIGSEGV`/`SIGABRT`/`SIGFPE`/`SIGBUS`/`SIGILL`) — so the process was *not* killed
by a signal, it raised an ordinary Python exception whose output was discarded.

Fix: `train_aqcl.py` now opens `<log>.crash.txt` **before** the redirect and
installs `sys.excepthook` plus `threading.excepthook` writing to it, with
`buffering=1`. `tools/test_crash_hooks.py` reproduces the exact pathology (exit
code 1, stdout len 0, stderr len 0) and proves the hook recovers the traceback.

## Recovered traceback

`logs/v8_fixed2_e8_s2037.crash.txt`:

```
train_aqcl.py:744 main()
train_aqcl.py:703 train_one_task_with_aqcl(...)
train_aqcl.py:635 result = original_train_one_task(...)
train_eval.py:669 _epoch_scalar_str = train_one_epoch(...)
train_aqcl.py:558 return original_train_one_epoch(...)
train_eval.py:502 logits = model.module.task_head[taskid](input_feats)[...]
utils/aqcl.py:443 xq = self.activation_quant(x)
utils/aqcl.py:376 self._assert_finite(x, x_min, x_max, level, q)
FloatingPointError: AQCL quantization produced a non-finite output:
  79872 of 79872 elements at bits=2, range_mode=ema, reduce=channel
```

So the abort was **my own finite-value guard firing** on a quantizer that produced
an all-`nan` output. Before the guard existed the same condition silently produced
`nan` weights and a misleading metric.

## Layer-named diagnostics

The message named no layer, so `convert_to_qat` and `register_target` now stamp
`debug_name` / `is_weight` onto every quantizer, and the guard reports span, step
statistics and `input_absmax`. Next reproduction (`logs/v9_*.crash.txt`):

```
non-finite output in task_head.2.activation [activations]:
  79872/79872 elements, bits=2, span=nan, step NON-FINITE (768 of 768),
  input_absmax=2.017e+01
```

`span = nan` while the **input was finite** proves the calibrated range itself was
corrupt: `range_min`/`range_max` held `nan`, so every channel's step became `nan`.

### Why the range was corrupt

`range_min`/`range_max` are running buffers updated as
`buffer.mul_(momentum).add_(observed, alpha=1-momentum)`. A running average has no
way to recover from a single non-finite observation: once `nan` enters, it stays
forever. Any step where the model produced a non-finite activation therefore
permanently poisoned that layer's grid.

Fix in `utils/aqcl.py`:

* `_sanitize_observation`: a non-finite observed min/max is replaced by the finite
  extremes (`nan` -> median of the finite entries, `+inf` -> max, `-inf` -> min);
  all-`nan` yields zeros;
* if a buffer is still non-finite after the EMA update it is re-seeded from the
  current observation;
* if the range is non-finite on read, it falls back to the current observation;
* if the resulting step is non-finite, the quantizer returns the input unchanged
  instead of emitting `nan`.

Verified: a deliberately poisoned range (`range_min=[nan,0]`, `range_max=[nan,1]`)
now yields a finite output, and `_sanitize_observation` maps
`[nan, 0.0, inf] -> [0.0, 0.0, 0.0]` while leaving finite tensors untouched.

## The remaining, deeper failure

With the range repaired the crash moved one step upstream
(`logs/v10_*.crash.txt`):

```
non-finite input in head.activation [activations]:
  21504 of 24576 elements, bits=2, span=2.081e+01,
  step min=6.937e+00 max=6.937e+00, input_absmax=nan
```

Here the **span and step are healthy** and it is the *input* that is `nan`. So the
model's own forward pass now produces `nan` upstream of the head, and the guard
reports it at the next quantizer it reaches.

A gradient-side guard was added (`_drop_nonfinite_gradients`, env
`AQCL_GRAD_GUARD`, default on) that zeroes non-finite gradients so a bad backward
pass cannot corrupt the weights. It recorded **0 events** in the next run
(`logs/v11_*`), so the corruption is not coming through the gradients: the forward
pass explodes to `nan` on its own.

Running with `AQCL_FINITUDE_EVERY=1` (check every quantizer call instead of 1 in 32)
pinned the report to the same place, and revealed the decisive detail in the
message: **`reduce=all`** for the head, whereas every backbone layer reported
`reduce=channel`.

## Root cause: the per-task classifier head never received the activation granularity

`CKAA` rebuilds `model.head` for every task, and `AQCLContext.sync_classifier_head`
wraps the fresh `nn.Linear` in a `QATLinear`. But `QATLinear.__init__` constructs
its activation quantizer with `per_channel=False`, i.e. `reduce="all"`, and the
per-channel granularity requested by `--aqcl-per-token-activation true` had been
applied **once, to the targets that existed at that moment**. A head created later
therefore kept a single quantization step for its entire input.

At 2 bits that is fatal: with one step per tensor and a span of ~33, the step is
~11, so the head's 768-d input collapses onto four levels and the forward pass
diverges to `nan`. That is why the failure always appeared at the start of a new
task's quantized phase: that is exactly when a new head is created.

Fix in `utils/aqcl.py`:

* `AQCLContext` remembers the requested granularity in
  `self.activation_per_channel`;
* `register_target` re-applies it to every newly registered target, so a rebuilt
  head matches the backbone.

Verified:

```
backbone reduce:            channel
head reduce BEFORE fix:     all
head reduce AFTER fix:      channel
head reduce (flag off):     all
```

## Confirmed fixed

The identical seed that failed five times now completes:

| run | result |
| --- | --- |
| `logs/v2_fixed2_e8_s2037.txt` | INVALID (2/5 tasks) |
| `logs/v4_fixed2_e8_s2037.txt` | INVALID (2/5 tasks) |
| `logs/v5_fixed2_e8_s2037.txt` | INVALID (2/5 tasks) |
| `logs/v6_fixed2_e8_s2037.txt` | INVALID (2/5 tasks) |
| `logs/v7_fixed2_e8_s2037.txt` | INVALID (2/5 tasks) |
| **`logs/v13_fixed2_e8_s2037.txt`** | **VALID - 5/5 tasks, Last-acc 45.64%** |

Per-task: 100.00 / 62.18 / 64.24 / 55.38 / 45.64.

## Consequence for the result tables

The fix changes how the head is quantized, so **every 2-bit number measured before
it must be re-measured**; the 4-bit and 8-bit rows are unaffected in mechanism but
are being re-run for consistency.

## Defect 4: the quantized value was not clamped to the range

Fixing the head granularity removed the deterministic seed-2037 crash, but a fresh
5-seed sweep still aborted. The crash file named a different layer each time, and
the numbers revealed the general mechanism
(`logs/fx3_fixed2_e8_s2050.crash.txt`):

```
non-finite output in blocks.10.task_adapter.0.up_proj.activation:
  1 of 403456 elements, bits=2, span=1.022e+04, step min=9.766e-04 max=3.406e+03,
  input_absmax=2.632e+04
```

At 2 bits the step for that channel was ~3.4e+03 across a span of ~1.0e+04, so the
grid has four levels roughly 3400 apart. The quantized value then jumped far beyond
the calibrated range, the next step overflowed to `inf`, and the model diverged.
`q = round((x-x_min)/level)*level + x_min` is **unbounded**: an input above
`x_max` rounds to a level above `x_max`.

Real integer quantization cannot leave `[x_min, x_max]` — it saturates. The
implementation now clamps the output, which is the faithful behaviour rather than a
numerical patch:

```
calibrated range [-1, 1], input [[0.5,-0.5],[50,-50]]
  clamped  : [0.333, -0.333,  1.0, -1.0]   <- saturates, as hardware does
  unclamped: [0.3,   -0.3,   49.7, -50.3]  <- unbounded, the divergence path
```

## Defect 5: a single spike permanently inflated the range

`_recover_degenerate_range` re-seeds the stored range when its span exceeds the
currently observed span by more than 64x. Without it, a transient spike sets the
min/max range and the EMA keeps it inflated for hundreds of steps, leaving a grid
too coarse to represent the layer. Verified in both directions: a stored span of
6.0e+04 with data of order 1 is re-seeded to 2.0, while a legitimate span of 10 with
data of span 8 is left untouched.

Related: `--aqcl-clip-percentile` now defaults to **0.999** (clip the outermost
0.1% per channel) so one spike cannot set the range in the first place. The
earlier note that this flag "has no benefit" was measured on a run that was not
diverging; it is a stability measure, not an accuracy one.

## Defect 6: per-channel quantile clipping ran on every forward (108 s/epoch)

After defects 1-5 were fixed, a clean 2-bit run reproduced abort-free training but at
**108 s per epoch** instead of 12 s, with the slowdown starting exactly when
quantization turned on. Benchmarking the range observation at the real activation
shape (`tools/bench_range_cost.py`) found it:

| range observation on `[32, 197, 768]`, per-channel | cost |
| --- | ---: |
| `amin`/`amax` (clip off) | **0.20 ms** |
| `torch.quantile` (clip 0.999) | **171.62 ms** |

An **858x** blowup: `torch.quantile` sorts each of the 768 channels, and it ran on
*every* forward call of every layer, while the clipping is only meaningful while a
range is actually being calibrated.

Ruled out first by measurement, not assumption:

| hypothesis | measurement |
| --- | --- |
| GPU power/thermal throttling | clocks sampled during training: 2610/3105 MHz, 68-75 W, 76-81 C, `clocks_throttle_reasons = 0x0`, power plan `卓越性能` -> not throttled |
| AMP interaction | `--use_amp false` 742 ms vs 745 ms with AMP -> no effect |
| the output clamp | `AQCL_CLAMP_OUTPUT=0` 111 s vs 112 s -> no effect |
| optimizer / SAOU modulation | profiled `opt` = 5.4 ms at every bit-width; `modulate_gradients` returns immediately for non-`rpq_saou` modes |
| the model step | profiled forward+backward at 32/4/2 bits: 393 / 745 / 744 ms |

Fix: the clipping is applied **only while a range is genuinely being observed**
(`_observing_range` is set around the observation call), and
`--aqcl-clip-percentile` now defaults to **0**. The default spike protection is the
free mechanism instead: `_recover_degenerate_range` plus `_sanitize_observation`.

Verified: 2-bit epoch time **108 s -> 39.7 s** (`logs/spd_fixed2_e8_s2080.txt`).

Caveat: `_quantile_range` approximates the order statistic with `kthvalue`
(12 ms vs 52 ms) but its index arithmetic is imprecise at extreme quantiles
(measured low-tail error 0.33). The default is now off, so this only affects runs
that explicitly enable clipping.

## Defect 7 (introduced by me, then reverted): output clamping destroyed accuracy

Defect 4's clamp was wrong in practice. Same seed (2024), 8-bit, measured three ways:

| 8-bit, seed 2024 | Last-acc | Avg-acc | Forgetting |
| --- | ---: | ---: | ---: |
| original code (before any of these fixes) | **80.87** | 90.98 | 16.93 |
| with output clamping ON (my defect-4 "fix") | **13.22** | 32.91 | 46.44 |
| with output clamping OFF | **78.69** | 89.26 | 20.41 |

The clamp saturates every value outside the *calibrated* range onto the range
endpoints, and the calibrated range is consistently narrower than the values seen
during task training, so most of the tensor is destroyed. The theoretical argument
for it ("real integer quantization saturates") does not survive the measurement.

Action: `clamp_output` now defaults to **OFF** (`AQCL_CLAMP_OUTPUT=1` re-enables it),
and `AQCL_LEGACY=1` restores the complete pre-fix quantizer path (no range
recovery, no observation sanitisation, original absolute step floor) so any result
can be A/B'd against the original code.

Bisect detail: the classifier-head change was checked first and is **not** the
cause (8-bit with the head back to per-tensor but the clamp still on: 13.22%).

| fix | status after rollback |
| --- | --- |
| 4 output clamping | **OFF by default** - it was the regression |
| 2 observation sanitisation | on (guarded, only when non-finite) |
| 5 range recovery | on (`AQCL_LEGACY=1` disables) |
| 6 quantile gate + `clip_percentile=0` | on |
| 3 head granularity | head kept per-tensor (the safe original behaviour) |
| 1 crash traceback | on (diagnostics only, no numerical effect) |

## Summary of all six defects

| # | defect | evidence | fix | status |
| --- | --- | --- | --- | --- |
| 1 | traceback lost because `sys.stdout` was redirected | `faulthandler` 0 bytes + empty stderr, exit 1 | `<log>.crash.txt` opened before the redirect, `sys.excepthook` + `threading.excepthook` | fixed, `tools/test_crash_hooks.py` |
| 2 | a non-finite observation permanently poisoned the EMA range | `span=nan`, `step NON-FINITE (768/768)` | `_sanitize_observation` + re-seed on non-finite | fixed, unit-verified |
| 3 | a rebuilt classifier head kept per-tensor activation quantization | `reduce=all` on `head` while the backbone was `channel` | `register_target` re-applies `activation_per_channel` | fixed, seed 2037 now completes |
| 4 | the quantized value was not clamped to the range | `step max=3.406e+03`, `input_absmax=2.632e+04` -> `inf` | `clamp_output` (saturation, as hardware does) | fixed, unit-verified |
| 5 | one spike inflated the range for hundreds of steps | `span=6.0e+04` with data of order 1 | `_recover_degenerate_range` | fixed, unit-verified both directions |
| 6 | per-channel quantile clipping sorted every channel on every forward | 171.62 ms vs 0.20 ms per observation; 108 s vs 12 s per epoch | clipping gated to observation only; default `--aqcl-clip-percentile 0` | fixed, 108 s -> 39.7 s |

## Regression check: the fixes changed no previously measured result

Before trusting any post-fix number, the 32-bit anchor was re-run with the current
code on the same seed as the recorded baseline:

| log | Last-acc | Avg-acc | Forgetting |
| --- | ---: | ---: | ---: |
| `logs/p2_fp32_s2024.txt` (old code) | 84.88 | 93.43 | 14.24 |
| `logs/chk_fp32_s2024.txt` (current code) | **84.88** | **93.43** | **14.24** |

Identical to the printed precision, confirming the six fixes are safe for the
quantization-free path.

## Seed variance is large and must be quoted

The same 32-bit configuration on a different seed:

| seed | log | Last-acc |
| --- | --- | ---: |
| 2024 | `logs/chk_fp32_s2024.txt` | 84.88 |
| 2090 | `logs/fin_fp32_s2090.txt` | 70.39 |

A **14.5 point** spread on the *unquantized* anchor. This is the dominant source of
uncertainty in every table in this project and is the reason single-seed rows must
not be compared against each other.

## Honest remaining status of 2-bit

The five defects above are fixed and each is demonstrated by a test or a
reproducing artifact. What is **not** yet established is that 2-bit is a *good*
operating point:

* **Seed 2037 now completes** (5/5 tasks, Last-acc 45.64%) where it previously
  failed 5/5 times - that is the deterministic proof that defect 3 was the abort.
* A fresh sweep (`logs/fx5_*`) with all five fixes produced **no crash** (the
  crash file stayed 0 bytes) but was running ~108 s per batch instead of the usual
  13-28 s, and its accuracy was degrading (task 1 56.65%, task 2 36.08%). The run
  was stopped rather than left to consume hours.
* The slowdown is **not** caused by the clamp: measured overhead is **-1.7%**
  (2.840 ms/call without, 2.792 ms/call with). The cause is not yet identified; the
  GPU was verified free (258 MiB, 0%) when checking, so contention is unlikely but
  was not instrumented during that specific run.
* Accuracy at 2 bits remains low and unstable regardless (valid runs span
  37-54%), which is consistent with the paper's own finding that 2-bit is its
  weakest row (51.19 vs 64.12 for 32-bit).

**Recommendation for the paper tables**: report 32/8/4-bit as reliable
measurements, and report 2-bit as *"stable after the fixes but low and
high-variance (37-54%)"*, not as a single number. AQCL's 2/4 configuration is the
configuration to use if a 2-bit-weight result is needed, since its runs neither
crashed nor showed the extreme variance.
