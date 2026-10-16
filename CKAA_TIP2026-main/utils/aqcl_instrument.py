"""Phase-1 instrumentation for AQCL (logging only, no behavior change).

This module is imported by `utils/aqcl.py` and `train_aqcl.py`. It only adds
observability: it never mutates quantizer state, bit allocation, gradients,
subspaces or optimizer behavior.

Design constraints:
  * pure PyTorch / stdlib, no third-party quantization framework
  * no training-only tensors are written into deployment checkpoints
  * every emitted record carries the source stage so a conclusion can always
    be traced back to a log line
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch


def _is_finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def summarize_scalar_tensor(values: torch.Tensor) -> dict[str, Any]:
    """Return robust min/max/mean/std/quantiles for a 1-D float tensor."""
    flat = values.detach().to(torch.float64).reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {
            "count": 0,
            "sum": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "num_zero": 0,
            "num_positive": 0,
        }
    q = torch.quantile(
        flat,
        torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
    )
    return {
        "count": int(flat.numel()),
        "sum": float(flat.sum().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "q25": float(q[0].item()),
        "q50": float(q[1].item()),
        "q75": float(q[2].item()),
        "num_zero": int((flat == 0).sum().item()),
        "num_positive": int((flat > 0).sum().item()),
    }


class AQCLInstrumentation:
    """Collects per-task instrumentation records for AQCL."""

    def __init__(self, enabled: bool = False, verbose: bool = True) -> None:
        self.enabled = bool(enabled)
        self.verbose = bool(verbose)
        self.records: list[dict[str, Any]] = []
        self.stage_counts: dict[str, int] = {}

    # ------------------------------------------------------------------ util
    def bump(self, stage: str) -> int:
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
        return self.stage_counts[stage]

    def log(self, stage: str, message: str) -> None:
        if not self.enabled or not self.verbose:
            return
        print(f"[AQCL-INSTRUMENT][{stage}] {message}", flush=True)

    # ------------------------------------------------------- sensitivity map
    def sensitivity_report(
        self,
        stage: str,
        taskid: int,
        weight_sensitivity: dict[str, torch.Tensor],
        activation_sensitivity: dict[str, torch.Tensor],
        weight_bits: dict[str, int],
        activation_bits: dict[str, int],
        low_bits: int,
        high_bits: int,
        weight_shapes: dict[str, tuple[int, ...]] | None = None,
        weight_numel: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Per-layer Fisher sensitivity vs assigned bit-width."""
        rows: list[dict[str, Any]] = []
        for name in weight_sensitivity:
            w = float(weight_sensitivity[name].detach().to(torch.float64).item())
            a_tensor = activation_sensitivity.get(name)
            a = (
                float(a_tensor.detach().to(torch.float64).item())
                if a_tensor is not None
                else float("nan")
            )
            rows.append(
                {
                    "layer": name,
                    "weight_sensitivity": w,
                    "activation_sensitivity": a,
                    "weight_bits": int(weight_bits.get(name, -1)),
                    "activation_bits": int(activation_bits.get(name, -1)),
                    "weight_shape": list(weight_shapes[name]) if weight_shapes and name in weight_shapes else None,
                    "weight_numel": int(weight_numel[name]) if weight_numel and name in weight_numel else None,
                    "low_bits": int(low_bits),
                    "high_bits": int(high_bits),
                }
            )

        w_values = torch.tensor(
            [row["weight_sensitivity"] for row in rows], dtype=torch.float64
        )
        a_values = torch.tensor(
            [row["activation_sensitivity"] for row in rows], dtype=torch.float64
        )
        w_stats = summarize_scalar_tensor(w_values)
        a_stats = summarize_scalar_tensor(a_values)

        w_bits_dist: dict[str, int] = {}
        for row in rows:
            key = str(row["weight_bits"])
            w_bits_dist[key] = w_bits_dist.get(key, 0) + 1
        a_bits_dist: dict[str, int] = {}
        for row in rows:
            key = str(row["activation_bits"])
            a_bits_dist[key] = a_bits_dist.get(key, 0) + 1

        degeneracy: list[str] = []
        if w_stats["num_zero"] == len(rows) and rows:
            degeneracy.append("all_weight_sensitivity_zero")
        if w_stats["std"] == 0.0 and len(rows) > 1:
            degeneracy.append("all_weight_sensitivity_equal")
        if a_stats["count"] and a_stats["num_zero"] == a_stats["count"]:
            degeneracy.append("all_activation_sensitivity_zero")
        non_low = sum(
            count for bits, count in w_bits_dist.items() if int(bits) != low_bits
        )
        nonzero = sum(
            count for bits, count in w_bits_dist.items() if int(bits) > 0
        )
        if len(rows) > 1 and non_low in (0, nonzero):
            degeneracy.append("weight_bits_not_mixed")
        record = {
            "stage": stage,
            "taskid": int(taskid),
            "mode": None,
            "low_bits": int(low_bits),
            "high_bits": int(high_bits),
            "num_layers": len(rows),
            "weight_sensitivity_stats": w_stats,
            "activation_sensitivity_stats": a_stats,
            "weight_bits_distribution": w_bits_dist,
            "activation_bits_distribution": a_bits_dist,
            "threshold_weight_mean": w_stats["mean"],
            "threshold_activation_mean": a_stats["mean"],
            "degeneracy_flags": degeneracy,
            "layers": rows,
        }
        self.records.append(record)

        self.log(
            stage,
            f"task={taskid} layers={len(rows)} "
            f"w_sens[min={w_stats['min']:.3e} max={w_stats['max']:.3e} "
            f"mean={w_stats['mean']:.3e} std={w_stats['std']:.3e} "
            f"zeros={w_stats['num_zero']}] "
            f"a_sens[min={a_stats['min']:.3e} max={a_stats['max']:.3e} "
            f"mean={a_stats['mean']:.3e} zeros={a_stats['num_zero']}] "
            f"w_bits={w_bits_dist} a_bits={a_bits_dist} "
            f"degenerate={degeneracy}",
        )
        if self.enabled and self.verbose:
            for row in rows:
                print(
                    "[AQCL-INSTRUMENT][layer] "
                    f"task={taskid} {row['layer']} "
                    f"shape={row['weight_shape']} "
                    f"w_sens={row['weight_sensitivity']:.6e} "
                    f"a_sens={row['activation_sensitivity']:.6e} "
                    f"w_bits={row['weight_bits']} a_bits={row['activation_bits']}",
                    flush=True,
                )
        return record

    # ------------------------------------------------------- subspace report
    def subspace_report(
        self,
        stage: str,
        taskid: int,
        eigenvalues: dict[str, torch.Tensor],
        subspaces: dict[str, dict[str, torch.Tensor]],
        covariance: dict[str, torch.Tensor],
        covariance_count: dict[str, int],
        lambda_threshold: float,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for name, values in eigenvalues.items():
            vals = values.detach().to(torch.float64)
            subspace = subspaces.get(name, {})
            uo = subspace.get("Uo")
            up = subspace.get("Up")
            total = int(vals.numel())
            uo_dim = int(uo.shape[1]) if uo is not None and uo.ndim == 2 else 0
            up_dim = int(up.shape[1]) if up is not None and up.ndim == 2 else 0
            cov = covariance.get(name)
            rows.append(
                {
                    "layer": name,
                    "feature_dim": total,
                    "covariance_count": int(covariance_count.get(name, 0)),
                    "eig_min": float(vals.min().item()) if total else float("nan"),
                    "eig_max": float(vals.max().item()) if total else float("nan"),
                    "eig_median": float(vals.median().item()) if total else float("nan"),
                    "eig_sum": float(vals.sum().item()) if total else float("nan"),
                    "eig_first": float(vals[0].item()) if total else float("nan"),
                    "eig_last": float(vals[-1].item()) if total else float("nan"),
                    "eig_rank_1e6": int((vals > 1e-6).sum().item()),
                    "eig_rank_1e3": int((vals > 1e-3).sum().item()),
                    "threshold_abs": float(
                        (lambda_threshold * vals[-1].clamp_min(1e-12)).item()
                    )
                    if total
                    else float("nan"),
                    "Uo_dim": uo_dim,
                    "Up_dim": up_dim,
                    "Uo_frac": (uo_dim / total) if total else 0.0,
                    "Up_frac": (up_dim / total) if total else 0.0,
                    "uo_numel": int(uo.numel()) if uo is not None else 0,
                    "cov_diag_mean": float(
                        torch.diagonal(cov).abs().mean().item()
                    )
                    if cov is not None and cov.ndim == 2
                    else float("nan"),
                    "cov_trace": float(torch.trace(cov).item())
                    if cov is not None and cov.ndim == 2
                    else float("nan"),
                    "cov_is_finite": bool(
                        torch.isfinite(cov).all().item()
                    )
                    if cov is not None
                    else False,
                }
            )
        record = {
            "stage": stage,
            "taskid": int(taskid),
            "lambda_threshold": float(lambda_threshold),
            "num_layers": len(rows),
            "layers": rows,
        }
        self.records.append(record)
        self.log(
            stage,
            f"task={taskid} layers={len(rows)} "
            + "; ".join(
                f"{r['layer']}: eig[{r['eig_min']:.3e},{r['eig_max']:.3e}] "
                f"thr={r['threshold_abs']:.3e} Uo={r['Uo_dim']}/{r['feature_dim']} "
                f"Up={r['Up_dim']} cov_n={r['covariance_count']}"
                for r in rows[:4]
            )
            + (" ..." if len(rows) > 4 else ""),
        )
        return record

    # --------------------------------------------------- gradient modulation
    def gradient_report(
        self,
        stage: str,
        taskid: int,
        epochs: dict[int, int],
        stats: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for name, values in stats.items():
            steps = values.get("steps", 0.0)
            row = {"layer": name, **values}
            if steps > 0:
                row["cos_step_mean"] = values.get("cos_sum", 0.0) / steps
                row["cos_sq_step_mean"] = values.get("cos_sq_sum", 0.0) / steps
                row["g_norm_step_mean"] = values.get("g_norm_sum", 0.0) / steps
                row["gp_norm_step_mean"] = values.get("gp_norm_sum", 0.0) / steps
                row["go_norm_step_mean"] = values.get("go_norm_sum", 0.0) / steps
                row["gamma_step_mean"] = values.get("gamma_sum", 0.0) / steps
                row["modulated_norm_step_mean"] = (
                    values.get("modulated_norm_sum", 0.0) / steps
                )
                row["applied_frac"] = values.get("applied_steps", 0.0) / steps
                # Alg. 2 line 2: |cos(g, Uo_perp)|^2 > theta.
                row["triggered"] = bool(
                    row["cos_sq_step_mean"] > values.get("theta", 0.5)
                )
            else:
                row["cos_step_mean"] = float("nan")
                row["cos_sq_step_mean"] = float("nan")
                row["applied_frac"] = 0.0
                row["triggered"] = False
            rows.append(row)

        def _stat(field: str) -> dict[str, Any]:
            values = [
                r[field]
                for r in rows
                if r.get("steps", 0.0) > 0
                and isinstance(r.get(field), float)
                and math.isfinite(r[field])
            ]
            tensor = torch.tensor(values, dtype=torch.float64) if values else torch.empty(0)
            return summarize_scalar_tensor(tensor)

        cos_stats = _stat("cos_step_mean")
        gamma_stats = _stat("gamma_step_mean")
        triggered = sum(1 for r in rows if r.get("triggered"))
        record = {
            "stage": stage,
            "taskid": int(taskid),
            "optimizer_steps_observed": int(epochs.get(0, 0)),
            "num_layers_modulated": len(rows),
            "cosine_step_mean_stats": cos_stats,
            "gamma_step_mean_stats": gamma_stats,
            "saou_triggered_layers": int(triggered),
            "saou_skipped_layers": int(len(rows) - triggered),
            "layers": rows,
        }
        self.records.append(record)
        self.log(
            stage,
            f"task={taskid} modulated_layers={len(rows)} "
            f"steps={record['optimizer_steps_observed']} "
            f"cos_step_mean[min={cos_stats['min']:.4f} mean={cos_stats['mean']:.4f} "
            f"max={cos_stats['max']:.4f}] "
            f"gamma_step_mean[min={gamma_stats['min']:.4f} "
            f"mean={gamma_stats['mean']:.4f}] "
            f"triggered={record['saou_triggered_layers']} "
            f"skipped={record['saou_skipped_layers']}",
        )
        if self.enabled and self.verbose:
            for row in sorted(
                rows, key=lambda r: r.get("cos_step_mean", 0.0), reverse=True
            )[:5]:
                print(
                    f"[AQCL-INSTRUMENT][saou_layer] task={taskid} {row['layer']} "
                    f"steps={int(row.get('steps', 0))} "
                    f"cos_mean={row['cos_step_mean']:.4f} "
                    f"cos_max={row.get('cos_max', float('nan')):.4f} "
                    f"cos_last={row.get('cos_last', float('nan')):.4f} "
                    f"applied_frac={row.get('applied_frac', 0.0):.3f} "
                    f"gamma_mean={row.get('gamma_step_mean', float('nan')):.4f} "
                    f"g_norm_mean={row.get('g_norm_step_mean', float('nan')):.3e} "
                    f"theta={row.get('theta', float('nan'))}",
                    flush=True,
                )
        return record

    # ------------------------------------------------------- bit-width budget
    def deployment_report(
        self,
        stage: str,
        taskid: int,
        weight_bits: dict[str, int],
        activation_bits: dict[str, int],
        model_size_mb: float,
        report: dict[str, Any],
        param_bits: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        w_counts: dict[str, int] = {}
        for bits in weight_bits.values():
            w_counts[str(bits)] = w_counts.get(str(bits), 0) + 1
        a_counts: dict[str, int] = {}
        for bits in activation_bits.values():
            a_counts[str(bits)] = a_counts.get(str(bits), 0) + 1
        param_counts: dict[str, int] = {}
        if param_bits:
            for bits in param_bits.values():
                param_counts[str(bits)] = param_counts.get(str(bits), 0) + 1
        record = {
            "stage": stage,
            "taskid": int(taskid),
            "num_quantized_layers": len(weight_bits),
            "weight_bit_layer_counts": w_counts,
            "activation_bit_layer_counts": a_counts,
            "weight_bits_by_layer": {k: int(v) for k, v in weight_bits.items()},
            "activation_bits_by_layer": {k: int(v) for k, v in activation_bits.items()},
            "all_parameter_bits": param_counts,
            "model_size_mb": float(model_size_mb),
            "gbops": float(report.get("inference_gbops", float("nan"))),
            "gflops": float(report.get("inference_gflops", float("nan"))),
            "gmac": float(report.get("inference_gmac", float("nan"))),
            "attention_mac": report.get("attention_mac"),
            "linear_mac": report.get("linear_mac"),
            "extra_memory_mb": float(
                report.get("inference_extra_memory_mb", float("nan"))
            ),
        }
        self.records.append(record)
        self.log(
            stage,
            f"task={taskid} size={model_size_mb:.4f}MB "
            f"w_bit_layers={w_counts} a_bit_layers={a_counts} "
            f"all_param_bits={param_counts} gbops={record['gbops']:.4f}",
        )
        if self.enabled and self.verbose:
            for name, bits in sorted(weight_bits.items()):
                print(
                    f"[AQCL-INSTRUMENT][alloc] task={taskid} {name} "
                    f"w={bits} a={activation_bits.get(name, -1)}",
                    flush=True,
                )
        return record

    # ------------------------------------------------------------------ dump
    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "instrumentation": "aqcl-phase1",
            "num_records": len(self.records),
            "stage_counts": self.stage_counts,
            "records": self.records,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"[AQCL-INSTRUMENT] wrote {path} ({len(self.records)} records)", flush=True)
