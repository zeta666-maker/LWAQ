# Comparison against the paper, and what can and cannot be fixed

## Measured (seed 2024, clean serial run, verified original code path)

| width | Last-acc | Avg-acc | retention | paper retention | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32-bit | 84.88 | 93.43 | 100.0% | 100.0% | - |
| 8-bit | 80.87 | 90.98 | 95.3% | 99.8% | **-4.5** |
| 4-bit | 68.85 | 83.94 | 81.1% | 99.6% | **-18.5** |
| 2-bit | (measuring) | | | 79.8% | |

Reproduce: `python tools/compare_to_paper.py --prefix F2024 --seed 2024`

Paper (AQCL, CIFAR-100, ResNet-20, T=10): 32-bit 64.12, 8-bit 63.96, 4-bit 63.84,
2-bit 51.19; retention 99.8% / 99.6% / 79.8%.

## Why the absolute numbers are not comparable (and must not be)

The paper's setting differs in the one way that decides the outcome:

| | paper | this project |
| --- | --- | --- |
| backbone | ResNet-20, **trained from scratch** | ViT-Base/16, **frozen** |
| trainable parameters | all | 2.15 M of 91.48 M (2.35%, at any time) |
| tasks / classes | 10 / 100 | 5 / 9 |
| epochs per task | 80 | 8 |

The paper's central claim - 4-bit within 0.28 points of 32-bit - is only achievable
because **every layer adapts to the quantization grid during training**. Here 98% of
the parameters (the 85.2 M backbone) are frozen and receive low-bit truncation that the
remaining 2.35% cannot compensate. That single structural fact, not a code defect,
accounts for the 4-bit gap.

## Is the 4-bit gap fixable without breaking the constraints?

Tested directly (earlier measurements in this project), holding the trainable set at
4-bit and raising only the **frozen backbone** floor:

| 4-bit configuration | Last-acc | size | note |
| --- | ---: | ---: | --- |
| whole network 4-bit (paper-faithful) | 68.85 | 45.54 MB | current row |
| 2-bit weights / 8-bit backbone | 82.28 | 84.87 MB | `--aqcl-backbone-bits 8` |
| 4-bit + attention-only Hadamard rotation | 74.26 (3 seeds) | 45.54 MB | +4.25 pts, same size |

So the retention gap closes to ~97% **only by giving the frozen backbone more
precision**, which contradicts "a 4-bit deployment". There is no configuration that
is simultaneously a genuine whole-network 4-bit model and paper-level retention,
under a frozen backbone.

## Paper-grounded changes that were applied

1. `--aqcl-hadamard true --aqcl-hadamard-scope attention` for 4-bit: an orthogonal
   change of basis that flattens weight outliers. Verified to cut weight
   quantization error 59.99% -> 12.99% at 4-bit, costs no model size, and measured
   **+4.25 Last-acc at 4-bit** with forgetting 22.5% -> 15.8% (3 seeds). It is
   harmful at 2-bit (31.74% vs 39.06%) and diverges if applied to the MLP, so it is
   scoped to the attention projections.
2. RPQ's Eq.14/Eq.15 are computed separately, so weight and activation widths are
   decoupled (`--aqcl-act-low-bits` / `--aqcl-act-high-bits`).
3. The default path was reverted to the original code after my output-clamping
   change was measured to destroy accuracy (8-bit 80.87 -> 13.22). See
   `ACCURACY_REGRESSION_AND_REVERT.md`.

## Honest statement for the report

* The 32-bit and 8-bit rows are sound; 8-bit retention is 95.3% against the paper's
  99.8%.
* The 4-bit row is 18.5 points below the paper's retention, and the cause is the
  frozen backbone, demonstrated by the fact that raising the backbone floor to
  8-bit recovers retention to ~97% at the cost of doubling the model size.
* 2-bit is reported only as a distribution with its failure rate, never as a single
  number, because it varies by ~17 points between valid runs and aborted in ~46% of
  recorded attempts before the crash-traceback fix.
