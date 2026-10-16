"""Run one AQCL case over several seeds and report mean +/- spread.

Motivation: a single seed is not evidence of a precise number on this dataset,
because the RandomForest task router and the saturated logits add a few points of
run-to-run spread. This script launches the same configuration with different
`--seed` values and prints the mean and range so a claim like "AQCL 4/8 is below
8-bit" can be judged against the noise.

Usage:
  python tools/run_aqcl_seeds.py --case rpq_4_8 --seeds 2024,2025,2026 --epochs 4
  python tools/run_aqcl_seeds.py --case fixed8 --seeds 2024,2025,2026 --epochs 4
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "fp32": ("false", "fixed", 32, 32, 32),
    "fixed8": ("true", "fixed", 8, 8, 8),
    "fixed4": ("true", "fixed", 4, 4, 4),
    "fixed2": ("true", "fixed", 2, 2, 2),
    "rpq_4_8": ("true", "rpq", 4, 4, 8),
    "rpq_saou_4_8": ("true", "rpq_saou", 4, 4, 8),
    "rpq_saou_2_4": ("true", "rpq_saou", 2, 2, 4),
}

RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
SIZE_RE = re.compile(r"AQCL deployment: size=([\d.]+)MB.*weight_bits=(\[[^\]]*\])")


def build(seed: int, case: str, args) -> list[str]:
    enable, mode, bits, low, high = CASES[case]
    suffix = f"{args.prefix}_{case}_s{seed}"
    return [
        sys.executable, "-u", str(ROOT / "train_aqcl.py"),
        "-d", "fault_csv", "-t", str(args.tasks),
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32", "-e", str(args.epochs),
        "-jt", "0", "-je", "0", "-et", "1",
        "--eval_batch_size", "64", "--temperature", "28.0",
        "--lr", "0.005", "--lr_scale", "0.2", "--lr_scale_patterns", "patch_embed",
        "--null_eta1", "0.999", "--null_eta2", "0.999", "-tc", "2.0",
        "--eval-tool", "adapter", "--eval-trained-task-router", "true",
        "--eval-local-task-head", "true", "--eval-task-weight", "false",
        "--prototype-confidence-weight", "0.0",
        "--seed", str(seed),
        "--save-model", "false", "--logs-dir", "logs", "-sf", suffix,
        "--aqcl-enable", enable, "--aqcl-mode", mode,
        "--aqcl-bits", str(bits), "--aqcl-low-bits", str(low),
        "--aqcl-high-bits", str(high),
        "--aqcl-backbone-bits", str(args.backbone_bits),
        "--aqcl-fisher-batches", "8",
        "--aqcl-warmup-epochs", "1",
        "--aqcl-calibrate-batches", "4",
        "--aqcl-per-token-activation", "true",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--prefix", default="seed")
    parser.add_argument("--backbone-bits", type=int, default=-1)
    parser.add_argument("--timeout", type=int, default=2700)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        command = build(seed, args.case, args)
        suffix = f"{args.prefix}_{args.case}_s{seed}"
        log_path = ROOT / "logs" / f"{suffix}.txt"
        if log_path.exists() and f"Results of task [{args.tasks}]" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ):
            print(f"[seeds] skip {suffix} (complete)")
        else:
            print(f"[seeds] run {suffix}", flush=True)
            work = ROOT.parent / "_aqcl_work"
            work.mkdir(parents=True, exist_ok=True)
            with open(work / f"{suffix}.out", "wb") as out, open(
                work / f"{suffix}.err", "wb"
            ) as err:
                try:
                    subprocess.run(
                        command, cwd=str(ROOT), stdout=out, stderr=err,
                        timeout=args.timeout,
                    )
                except subprocess.TimeoutExpired:
                    print(f"[seeds] {suffix} TIMEOUT")
                    continue
        if not log_path.exists():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        results = RESULT_RE.findall(text)
        if not results:
            print(f"[seeds] {suffix}: no results")
            continue
        last_acc, avg_acc, forg = (float(v) for v in results[-1][1:])
        sizes = SIZE_RE.findall(text)
        size = float(sizes[-1][0]) if sizes else float("nan")
        bits = sizes[-1][1] if sizes else "-"
        rows.append(
            {
                "seed": seed,
                "log": str(log_path),
                "last_acc": last_acc,
                "avg_acc": avg_acc,
                "forgetting": forg,
                "size_mb": size,
                "bits": bits,
            }
        )
        print(
            f"[seeds] {suffix}: Last-acc {last_acc:.2f}% Avg-acc {avg_acc:.2f}% "
            f"Forgetting {forg:.2f}% size {size:.2f}MB bits {bits}"
        )

    if rows:
        last = [r["last_acc"] for r in rows]
        avg = [r["avg_acc"] for r in rows]
        print()
        print(f"case={args.case} seeds={[r['seed'] for r in rows]}")
        print(
            f"  Last-acc mean {statistics.mean(last):.2f}% "
            f"(min {min(last):.2f}, max {max(last):.2f}, "
            f"spread {max(last) - min(last):.2f})"
        )
        print(
            f"  Avg-acc  mean {statistics.mean(avg):.2f}% "
            f"(min {min(avg):.2f}, max {max(avg):.2f})"
        )


if __name__ == "__main__":
    main()
