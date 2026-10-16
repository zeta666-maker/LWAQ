# AQCL Implementation Notes (paper 2)

AQCL is implemented in `utils/aqcl.py` (quantization, RPQ, SAOU, deployment
report), instrumented by `utils/aqcl_instrument.py`, and driven by the separate
entry point `train_aqcl.py`. The CKAA entry point `train_eval.py` contains **no**
AQCL code; its last modification (2026‑09‑28 22:34) predates this work.

Paper: *Adaptive Quantization for Stable Knowledge Acquisition in
Quantization-Aware Continual Learning* (TCSVT, DOI 10.1109/TCSVT.2026.3715272).

## 1. Method mapping (paper equation → code location)

| Paper | Content | Code |
| --- | --- | --- |
| Eq. 1 | asymmetric uniform quantization `x_q = round((x-l)/n)*n + l`, `n=(u-l)/(2^b-1)` | `UniformQuantizer.forward` (per-tensor activations, per-output-channel weights), straight-through estimator |
| Eq. 2–9 | perturbation, Taylor expansion, trace identity | derivation only; result is Eq. 11 |
| Eq. 10–11 | layer sensitivity = normalized Hessian trace `S_l = Tr(H_l)/(2·n_l·d_l)` | `AQCLContext._fisher_and_covariance`: `weight_sums[name].sum() / numel` |
| Eq. 12 | `Tr(H) ≈ Tr(F)` diagonal empirical Fisher | `sum(grad²)` accumulated over `--aqcl-fisher-batches` batches |
| Eq. 13 | `b_l = b_H if S_l ≥ S_avg else b_L`, applied to weights and activations separately | `AQCLContext._allocate_bits` |
| Eq. 14–15 | separate weight / activation sensitivity | `weight_sensitivity` (from `weight.grad`) and `activation_sensitivity` (from the full-backward hook on module inputs) |
| Eq. 16 | `g = g_o + g_p` | `go = (g @ Uo) @ Uoᵀ`, `gp = g - go` |
| Eq. 17–19 | `Xᵀ_{t-1} g_t = 0`, `g_o = Uo Uoᵀ g` | `modulate_gradients` |
| Eq. 20 | `X_{t-1} = U Σ Uᵀ` | `torch.linalg.eigh` on the symmetrized accumulated covariance |
| Eq. 21–22 | `X_cur = ZᵀZ/n_t`, `X_t = (ñ_{t-1} X_{t-1} + n_t X_cur)/ñ_t` | `covariance_hook` + weighted accumulation in `_fisher_and_covariance` |
| Eq. 23 | `M_ij = exp(-α·cos²(g_p,u_pi))` for `i=j` | `mask = torch.exp(-alpha * cosine.pow(2))` |
| Eq. 24 | `g' = g_o + U_p M (U_p M)ᵀ g_p` | `go + (coefficients * mask) @ up.t()` |
| Alg. 2 line 2 | switch: `|cos(g, Ū_o)| > θ` | `g_p.norm() > theta * g.norm()` (equals the projected squared-cosine because `U_p` is kept orthonormal) |
| Alg. 2 line 8 | otherwise keep only `g_o` | gradient replaced by `go` |
| Alg. 1 / Table VII | deployment uses only quantized weights + bit-widths; no Fisher/SVD at inference | `deployment_state`, `model_size_mb`, `inference_report`; Fisher/covariance/subspaces live only in `AQCLContext` |

Paper hyper-parameters reproduced by default: `λ = 10` (`--aqcl-lambda`), `θ`
(`--aqcl-theta`, default 0.5), `α = 5` (`--aqcl-alpha`), `b_H = 2·b_L`
(`--aqcl-high-bits = 2 × --aqcl-low-bits`), transition subspace `[0.5λ, 2λ]`.

## 2. Commands

Instrumented AQCL run (5 tasks, 2/2/2/2/1, 2 epochs, 1 warm-up epoch):

```powershell
python -u .\train_aqcl.py -d fault_csv -t 5 -m vit_base_patch16_224.augreg_in21k `
  -b 32 -e 2 -jt 0 -je 0 -et 1 --eval_batch_size 64 --temperature 28.0 `
  --lr 0.005 --lr_scale 0.2 --lr_scale_patterns patch_embed `
  --null_eta1 0.999 --null_eta2 0.999 -tc 2.0 `
  --eval-tool adapter --eval-trained-task-router true --eval-local-task-head true `
  --eval-task-weight false --prototype-confidence-weight 0.0 --seed 2024 `
  --save-model false --logs-dir logs -sf aqcl_p4_rpq_saou_4_8 `
  --aqcl-mode rpq_saou --aqcl-low-bits 4 --aqcl-high-bits 8 `
  --aqcl-fisher-batches 2 --aqcl-warmup-epochs 1 `
  --aqcl-instrument true
```

Whole comparison matrix (PowerShell script execution is disabled on this
machine, so use the Python runner):

```powershell
python -u .\tools\run_aqcl_matrix.py --prefix p4 --instrument
python -u .\tools\run_aqcl_matrix.py --prefix p4_paperfaithful --backbone-bits 0 --instrument
```

Summarise every log with its evidence path:

```powershell
python .\tools\aqcl_summary.py --logs-dir logs --pattern "p4_*.txt"
python .\tools\aqcl_inspect_json.py logs\p4_rpq_saou_4_8.instrument.json --task 1 --stage rpq_sensitivity
python .\tools\aqcl_rpq_probe.py -d fault_csv -t 5 -b 32 --logs-dir logs -sf probe
python .\tools\aqcl_fisher_grad_probe.py --probe-cpu -d fault_csv -t 5 -b 8
```

## 3. Phase-1 instrumentation (logging only)

`--aqcl-instrument true` records, per task, into
`logs/<suffix>.instrument.json` and into the run log:

* `rpq_sensitivity`: per-layer weight/activation Fisher sensitivity, the
  threshold `S_avg`, the assigned weight/activation bit-width and the bit-width
  distribution, plus degeneracy flags.
* `saou_subspace`: per-layer `Uo`/`Up` dimensions, eigenvalue min/max/median,
  threshold `λ·σ_min`, effective rank, covariance sample count.
* `saou_gradient`: per-layer `cos(g, Ū_o)` mean/min/max/last, the trigger
  decision (`|cos|² > θ`), the per-layer gain `γ = ‖g'_p‖/‖g_p‖`, and how many
  optimizer steps were observed.
* `deployment_bits`: layer counts per bit-width, all-parameter bit histogram,
  model size, GFLOPs/GBOPS.
* `[logits]` and `[fisher]` lines: quantized vs FP32 logit amplification at
  identical weights, logit saturation fraction, and per-batch gradient
  existence/zero statistics.

Instrumentation never writes training-only tensors into a deployment checkpoint:
the telemetry buffers are `persistent=False` and `torch.save(model.state_dict())`
therefore contains only `bits` (documentation) plus real weights.

## 4. Verified findings on fault_csv

Protocol: `fault_csv`, 5 incremental tasks with class split 2/2/2/2/1, 2 epochs
per task, 1 warm-up epoch, batch size 32, seed 2024, single RTX 4060 Laptop.

### 4.1 RPQ previously degenerated to all-8-bit (confirmed and repaired)

Evidence (`logs/aqcl_p1_instr_rpq_saou_4_8.txt` and
`logs/dbg_t5_e2_task0.txt`):

```text
[fisher] task=0 batch=0 loss=0.000000 logits[absmax=120.562] ... grad_non_none=74/187 max_abs_grad=0.000e+00
[fisher_summary] task=0 targets=187 grad_None_layers=113 grad_all_zero_layers=74 grad_nonzero_layers=0
[rpq_sensitivity] task=0 layers=187 w_sens[min=0.000e+00 max=0.000e+00 mean=0.000e+00 std=0.000e+00 zeros=187]
                 w_bits={'8': 187} degenerate=['all_weight_sensitivity_zero', ...]
```

Root-cause chain:

1. Training drives the 2-class logits to ±120 (measured FP32 `absmax=121.81`
   at the identical weights, i.e. not a quantization artefact).
2. `CrossEntropyLoss` on such logits is exactly `0.0` and its gradient is
   exactly zero.
3. `S_l = mean(grad²) = 0` for every layer; `S_avg = 0`; Eq.13's `S_l ≥ S_avg`
   then returns "high bits" for **all 187 layers**.

Repair (in `_allocate_bits` / `_fisher_and_covariance`):

* `layer_alive[name]`: a layer only enters `S_avg` if it participated in the
  current-task forward pass (unused future task adapters/heads no longer bias
  the threshold and are assigned the low bit-width).
* If no alive layer has a non-zero signal, the allocator keeps high precision
  and records `bit_allocation_degenerate = true` instead of silently producing
  an all-high-bit "allocation".
* `sync_classifier_head`: `train_eval.modify_head` rebuilds `model.head` every
  task; the new `nn.Linear` is re-registered so the head stays inside the
  RPQ/SAOU scope (`AQCL_QUANTIZE_HEAD=0` disables this for ablation).

After the repair (probe + run evidence):

```text
weight sensitivity min=0.0 max=7.24e-01 mean=3.01e-02 std=9.20e-02  (not constant)
w_bits={'8': 15, '4': 72}   degenerate=[]
```

### 4.2 SAOU trigger statistics

`logs/p2_rpq_saou_4_8.instrument.json` (`saou_gradient` records):
`cos(g, Ū_o)` is ≈ 0.99 for modulated layers, i.e. `|cos|² > θ = 0.5`, so SAOU
takes the similarity-aware branch (Alg. 2 line 3) and not the conservative
branch. The modulation is therefore active, but it only reaches layers that have
both a gradient and stored subspaces — for frozen-backbone CKAA configurations
that is the task adapter/head/prompt set, not the frozen ViT blocks.

### 4.3 4-bit quantization of the frozen backbone destroys the representation

2-task ablation, identical seed and schedule (`logs/dbg2_nohead.*`,
`logs/dbg2_nobackbone.*`, `logs/p3_floor8_4_8.txt`):

| Quantization scope (task 1 / task 2) | Last-acc T1 | Last-acc T2 | Avg-acc T2 |
| --- | ---: | ---: | ---: |
| whole network, 4/8-bit (paper-faithful Eq.13) | 72.47% | 32.12% | 52.29% |
| head unquantized | 72.47% | 32.12% | 52.29% |
| backbone FP32, trainable set 4-bit (`AQCL_QUANTIZE_BACKBONE=0`) | 99.68% | 67.41% | 83.54% |
| backbone floor 8-bit, trainable set 4/8-bit (`--aqcl-backbone-bits 8`) | 99.68% | 79.59% | 89.64% |

The AQCL paper quantizes every trainable layer of a from-scratch ResNet-20. In
this repository the ViT backbone is frozen and ~86 M of its parameters are
pretrained; quantizing them to 4-bit destroys features that the small trainable
set (~1.2 M parameters) cannot recover. The default is therefore
`--aqcl-backbone-bits 8`: the frozen backbone is floored at 8-bit and Eq.13 still
produces a genuine 4/8-bit mixture inside the CKAA trainable set
(`logs/p3_floor8_smoke.instrument.json`: pre-floor RPQ `{'8': 6, '4': 181}`,
effective `{'8': 50, '4': 137}`, trainable-set `{4: 137, 8: 1}`). Use
`--aqcl-backbone-bits 0` for the paper-faithful whole-network split.

## 5. Results

See `AQCL_RESULTS.md` — every row is generated from the logs by
`tools/aqcl_summary.py`, and each conclusion is quoted with its log path.

## 6. Risks and open issues

* The training schedule is 2 epochs/task instead of the paper's 80, and the
  backbone/dataset differ (tabular ViT vs ResNet-20 on CIFAR-100).
* Logits saturate (±120 and up) during CKAA training on this dataset; many
  batches then contribute an exactly-zero Fisher gradient, so the sensitivity
  estimate is sparse. The allocator is now robust to this, but the estimate is
  still noisier than the paper's.
* `inference_report` counts Linear/Conv1d MACs; attention `QKᵀ` and
  `attention @ V` are still not counted, so GFLOPs/GBOPS remain incomplete.
* The trained RandomForest task router is an extension of this repository, not
  the paper's Eq.10–12 TC-MoA; its accuracy is reported separately per task and
  is a confounder for the absolute numbers.
* The severity of 4-bit damage to the frozen backbone is a property of this
  setup; the paper's ResNet-20 result is not contradicted by it, but the two are
  not directly comparable.
