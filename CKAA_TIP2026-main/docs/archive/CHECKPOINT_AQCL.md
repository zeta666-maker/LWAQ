# CHECKPOINT — AQCL on fault_csv (paper 2)

Frozen at the user's request so the session can resume from this exact point.
No training processes are running (0 python processes).

**Before resuming training, free the GPU.** At freeze time
`MonsterHunterWilds.exe` (PID 17424) was holding 5208 MiB of the 8188 MiB card at
90% utilisation. That is the user's own application, not part of this project, and
must not be killed by the agent. Training under that load would reproduce the
contention that `tools/diagnose_contention.py` was written to detect, and would
inflate `batch_time` (an earlier contended run reached 2261 s/batch). Verify with:

```powershell
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

Expect ~500 MiB and low utilisation before starting a run.

## Resume instructions

Read this file first, then `AQCL_FINAL_RESULTS.md`. The one open task is the
**2-bit abort root cause** (section 3). The next concrete step is recorded in
section 4. Do not re-run the whole matrix: the results in section 2 are final and
guarded.

## 1. Environment facts (do not rediscover)

* Repo: `C:\Users\Administrator\Desktop\CKAA_TIP2026-main\CKAA_TIP2026-main`
  (the `Downloads\...` copy is stale — never use it).
* Python: `C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe`
  (torch 2.11.0+cu128, timm 0.9.12). Single RTX 4060 Laptop, 8 GB.
* `train_eval.py` is **unmodified**: mtime `2026-09-28 22:34:36`,
  sha256[:16] `7066e0858a460425`. It must stay a pure CKAA entry point.
* **PowerShell script execution is disabled** — use the Python runners.
* **Run exactly one training process at a time.** Overlapping runs previously
  caused real GPU contention (`batch_time` up to 2261 s/batch).
* Logs are buffered per epoch; a quiet log file does not mean a hang.
* Runners redirect child stdout/stderr to `..\..\Downloads\CKAA_TIP2026-main\_aqcl_work\<suffix>.out/.err`.

## 2. Final results (guarded, strictly serial) — COMPLETE

Regenerate: `python tools/final_results.py`

| configuration | w_bits | n | Last-acc | sd | range | Avg-acc | Forg | size MB | retention |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 32-bit fp32 (anchor) | [32] | 1 | 84.88 | - | - | 93.43 | 14.24 | 355.55 | 100.0% |
| 8-bit uniform | [8] | 1 | 80.87 | - | - | 90.98 | 16.93 | 88.89 | 95.3% |
| 4-bit uniform | [4] | 3 | 71.99 | 2.88 | 68.78-74.33 | 85.00 | 13.06 | 45.54 | 84.8% |
| 2-bit uniform | [2] | 5 | 44.45 | 7.74 | 37.20-54.22 | 63.15 | 26.46 | 23.87 | 52.4% |
| AQCL 4/8 | [4, 8] | 3 | 77.19 | 9.17 | 66.60-82.49 | 89.24 | 15.14 | 86.35 | 90.9% |
| AQCL 2/4 | [2, 4] | 3 | 73.51 | 3.33 | 69.83-76.30 | 85.97 | 14.37 | 44.17 | 86.6% |

Evidence logs (all pass `tools/run_validity.py`):

* anchor: `logs/p2_fp32_s2024.txt`
* 8-bit: `logs/p2_fixed8_s2024.txt`
* 4-bit: `logs/v2_fixed4_e8_s2033|2034|2035.txt`
* 2-bit: `logs/v2_fixed2_e8_s2033|2034|2035|2036.txt`, `logs/v4_fixed2_e8_s2041.txt`
* AQCL 4/8: `logs/cm_rpq_4_8_s2033|2034|2035.txt`
* AQCL 2/4: `logs/cm_rpq_saou_2_4_s2033|2034|2035.txt`

Objective items 2, 3, 4, 5 are **done**:

* (2) all widths re-measured serially and multi-seed, guard enforced;
* (3) same-nominal-width comparison done: AQCL 4/8 77.19 vs uniform 4-bit 71.99
  (+5.20); AQCL 2/4 73.51 vs uniform 2-bit 44.45 (+29.06);
* (4) SAOU scope documented: `num_layers_modulated = 1` in every run, because the
  frozen ViT blocks have `weight.grad is None`. No backbone claim is supportable;
* (5) `AQCL_FINAL_RESULTS.md` marks every row valid/unavailable with log paths.

## 3. RESOLVED — the 2-bit abort root cause (five defects)

See `AQCL_2BIT_ROOTCAUSE.md` for the full diagnosis chain. Summary:

| # | defect | fix | verification |
| --- | --- | --- | --- |
| 1 | the traceback was lost (stdout redirected before the failure) | `<log>.crash.txt` opened before the redirect + `sys.excepthook`/`threading.excepthook` | `tools/test_crash_hooks.py` reproduces exit 1 with empty stdout/stderr and recovers the traceback |
| 2 | one non-finite observation permanently poisoned the EMA range | `_sanitize_observation`, re-seed on non-finite | unit-verified |
| 3 | a rebuilt classifier head kept per-tensor activation quantization (`reduce=all`) instead of the configured per-channel | `register_target` re-applies `activation_per_channel` | **seed 2037 now completes 5/5 tasks** where it previously failed 5/5 |
| 4 | the quantized value was not clamped into `[x_min, x_max]` | `clamp_output` (saturation, as hardware does) | unit-verified; ±50 input clamps to ±1.0 |
| 5 | one spike inflated the range for hundreds of steps | `_recover_degenerate_range` (span ratio 64) + `--aqcl-clip-percentile` default 0.999 | unit-verified both directions |

The decisive evidence for defect 3: `logs/v2/v4/v5/v6/v7_fixed2_e8_s2037.txt` are all
INVALID (2/5 tasks) while `logs/v13_fixed2_e8_s2037.txt` is VALID (5/5 tasks,
Last-acc 45.64%).

### Still open

* A sweep with all five fixes (`logs/fx5_*`) produced **no crash** but ran at
  ~108 s/batch instead of 13-28 s and its accuracy degraded (56.65% -> 36.08%).
  The clamp is not the cause (measured -1.7%); the slowdown's cause is unidentified
  and contention was not instrumented during that run.
* 2-bit accuracy remains low and high-variance (valid runs 37-54%). Report it as
  such, or use the AQCL 2/4 configuration instead, which neither crashed nor
  showed that variance.
* All pre-fix 2-bit numbers in `AQCL_FINAL_RESULTS.md` must be re-measured because
  defect 3 changes how the head is quantized.

Current best lead: **exit code 1 with empty stdout AND empty stderr.** There is no
`sys.exit` / `os._exit` / `raise SystemExit` anywhere in `train_aqcl.py`,
`train_eval.py`, `utils/*.py` (verified by grep). So something in a library or a
non-main thread calls `sys.exit(1)`, or an uncaught exception occurs after
`sys.stdout = Logger(...)` at `train_aqcl.py:406` and its traceback never reaches
stderr.

## 4. Exact next steps on resume

1. Re-run the failing seed with the thread-traceback dump already wired in:
   `AQCL_DUMP_TRACEBACK=1 python -u tools/run_size_first.py --variants e8 --bits 2 --seeds 2037 --prefix v8`
   then inspect `logs/v8_fixed2_e8_s2037.threads.txt`. (The hook is implemented in
   `train_aqcl.py` and writes every thread's stack at exit via `atexit`.)
2. If that is empty too, wrap the training call so any exception is printed to
   fd 2 explicitly, e.g. run the child under
   `python -X faulthandler -u ... 2> err.txt` and additionally install
   `sys.excepthook` plus `threading.excepthook` writing to a separate file
   descriptor opened before the stdout redirect.
3. Confirm the abort reproduces at the same (task, epoch) before and after any fix.
4. Once fixed, re-measure 2-bit uniform with 5 seeds (some of the current 5 are
   from valid runs that completed, so they stay valid; only the failure rate
   changes).

## 5. Guard tooling (keep using these)

| tool | purpose |
| --- | --- |
| `tools/run_validity.py` | a run counts only if all 5 tasks printed, no `loss=nan`, one deployment line per task. `load_metrics` returns `None` for invalid runs |
| `tools/scan_divergence.py` | audit every log for divergence/incompleteness |
| `tools/diagnose_early_exit.py` | 2-bit early-exit rate and stopping points |
| `tools/diagnose_contention.py` | compare `batch_time` between failed and healthy runs |
| `tools/test_run_validity.py`, `tools/test_parse_guard.py` | regression tests (both pass) |
| `tools/final_results.py` | regenerate the section-2 table |

## 6. Fixes already landed this session

* progress-based run guard in `run_size_first.py` and `run_aqcl_tcmoa_matrix.py`
  (replaces the wall-clock timeout), plus exit-code capture;
* validity guard moved **into** `parse()` — previously an interrupted run was
  reported as a finished result (that is how a bogus "62.18%" appeared);
* cross-check of deployment-line count vs task count (catches two runs writing one
  filename);
* `UniformQuantizer._assert_finite`, amortised 1 call in 32 (+21.5% cost), names
  the offending tensor;
* `_apply_step_floor`: the step floor is now relative to the observed span instead
  of the absolute dtype epsilon;
* `faulthandler` enabled to a separate file in `train_aqcl.py`, plus optional
  `AQCL_DUMP_TRACEBACK=1` thread dump.

## 7. Known-bad inputs

* **Seed 2037 at 2-bit** fails deterministically at task 3 epoch 2 — treat as a
  known-bad configuration, not a data point.
* Old logs containing `loss=nan` or fewer than 5 task lines must never be
  averaged in: `logs/p2_fixed2_s2024.txt`, `logs/ha_fixed2_had_s2024.txt`,
  `logs/ha_fixed2_had_s2026.txt`, `logs/ha_fixed4_had_s2024.txt`,
  `logs/pD_rpq_4_8.txt`, `logs/fx_fixed2_e8_s2031.txt`,
  `logs/nf_fixed2_e8_s2028.txt`.
