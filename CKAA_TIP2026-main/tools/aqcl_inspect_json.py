"""Inspect an AQCL instrumentation JSON dump.

Prints the per-task bit allocation, the layers that received the high bit-width,
the Fisher sensitivity spread, the Uo/Up dimensions, eigenvalue ranges and the
SAOU gradient statistics. Used to verify every RPQ/SAOU conclusion against the
recorded evidence instead of trusting the summary numbers.

Usage:
  python tools/aqcl_inspect_json.py logs/p2_rpq_saou_4_8.instrument.json --task 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--stage", default="")
    args = parser.parse_args()

    payload = load(args.json_path)
    print(
        f"file={args.json_path} records={payload.get('num_records')} "
        f"stages={payload.get('stage_counts')}"
    )

    for record in payload["records"]:
        if args.task is not None and record.get("taskid") != args.task:
            continue
        if args.stage and record.get("stage") != args.stage:
            continue
        stage = record.get("stage")
        task = record.get("taskid")
        if stage == "rpq_sensitivity":
            print(f"\n=== rpq_sensitivity task={task} ===")
            print(
                f"mode={record.get('mode')} bL={record.get('low_bits')} "
                f"bH={record.get('high_bits')} layers={record.get('num_layers')}"
            )
            print(
                f"weight sensitivity: {record['weight_sensitivity_stats']}"
            )
            print(
                f"activation sensitivity: {record['activation_sensitivity_stats']}"
            )
            print(f"weight bit distribution: {record['weight_bits_distribution']}")
            print(
                f"activation bit distribution: "
                f"{record['activation_bits_distribution']}"
            )
            print(f"thresholds: w={record.get('weight_threshold')} "
                  f"a={record.get('activation_threshold')}")
            print(f"degenerate={record.get('degeneracy_flags')} "
                  f"degenerate_flag={record.get('bit_allocation_degenerate')}")
            rows = record["layers"]
            high = [r for r in rows if r["weight_bits"] == record.get("high_bits")]
            print(f"-- layers with high weight bits ({len(high)}):")
            for row in sorted(high, key=lambda r: -r["weight_sensitivity"])[: args.top]:
                print(
                    f"   {row['layer']} shape={row.get('weight_shape')} "
                    f"w_sens={row['weight_sensitivity']:.6e} "
                    f"a_sens={row['activation_sensitivity']:.6e} "
                    f"w={row['weight_bits']} a={row['activation_bits']}"
                )
            print("-- top layers by weight sensitivity regardless of bits:")
            for row in sorted(
                rows, key=lambda r: -r["weight_sensitivity"]
            )[: args.top]:
                print(
                    f"   {row['layer']} w_sens={row['weight_sensitivity']:.6e} "
                    f"w={row['weight_bits']} a={row['activation_bits']}"
                )
        elif stage == "saou_subspace":
            print(f"\n=== saou_subspace task={task} layers={record.get('num_layers')} ===")
            for row in record["layers"][: args.top]:
                print(
                    f"   {row['layer']} dim={row['feature_dim']} "
                    f"eig[{row['eig_min']:.3e},{row['eig_max']:.3e}] "
                    f"thr={row['threshold_abs']:.3e} Uo={row['Uo_dim']} "
                    f"Up={row['Up_dim']} cov_n={row['covariance_count']}"
                )
        elif stage == "saou_gradient":
            print(f"\n=== saou_gradient task={task} ===")
            print(
                f"modulated={record.get('num_layers_modulated')} "
                f"steps={record.get('optimizer_steps_observed')} "
                f"triggered={record.get('saou_triggered_layers')} "
                f"skipped={record.get('saou_skipped_layers')}"
            )
            print(f"cos stats: {record.get('cosine_step_mean_stats')}")
            print(f"gamma stats: {record.get('gamma_step_mean_stats')}")
            for row in record["layers"][: args.top]:
                print(
                    f"   {row['layer']} steps={int(row.get('steps', 0))} "
                    f"cos_mean={row.get('cos_step_mean')} "
                    f"cos_min={row.get('cos_min')} cos_max={row.get('cos_max')} "
                    f"applied_frac={row.get('applied_frac')} "
                    f"gamma_mean={row.get('gamma_step_mean')} "
                    f"g_norm_mean={row.get('g_norm_step_mean')}"
                )
        elif stage == "deployment_bits":
            print(f"\n=== deployment_bits task={task} ===")
            print(
                f"size={record['model_size_mb']:.4f}MB gbops={record['gbops']:.4f} "
                f"gflops={record['gflops']:.4f}"
            )
            print(
                f"weight bit layer counts: {record['weight_bit_layer_counts']} "
                f"activation: {record['activation_bit_layer_counts']} "
                f"all params: {record['all_parameter_bits']}"
            )


if __name__ == "__main__":
    main()
