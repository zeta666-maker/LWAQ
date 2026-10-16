"""Compare the measured AQCL rows against the paper's Table I and flag the gaps.

Paper (AQCL, TCSVT, CIFAR-100, ResNet-20, T=10, ACC = mean per-task accuracy after
the final task):

    32-bit 64.12  8-bit 63.96  4-bit 63.84  2-bit 51.19
    retention vs 32-bit:  8-bit 99.8%  4-bit 99.6%  2-bit 79.8%

Our setting is different (fault_csv, 9 classes, 5 tasks, ViT-Base/16 tabular, 8
epochs/task, and crucially a FROZEN backbone with only 5.9 M of 91 M parameters
trainable), so only the retention ratios are meaningfully comparable.

Usage: python tools/compare_to_paper.py --seed 2024
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)

PAPER = {
    "32-bit": {"acc": 64.12, "retention": 100.0},
    "8-bit": {"acc": 63.96, "retention": 99.8},
    "4-bit": {"acc": 63.84, "retention": 99.6},
    "2-bit": {"acc": 51.19, "retention": 79.8},
}

CASES = [("32-bit", "fp32"), ("8-bit", "fixed8"), ("4-bit", "fixed4"), ("2-bit", "fixed2")]


def load(prefix: str, case: str, seed: int) -> dict | None:
    path = ROOT / "logs" / f"{prefix}_{case}_s{seed}.txt"
    quantized = case != "fp32"
    validity = check_run(path, expected_tasks=5, quantized=quantized)
    if not validity.ok:
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    results = RESULT_RE.findall(text)
    if len(results) < 5:
        return None
    return {
        "last": float(results[-1][1]),
        "avg": float(results[-1][2]),
        "forg": float(results[-1][3]),
        "log": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="F2024")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    rows = {}
    for label, case in CASES:
        record = load(args.prefix, case, args.seed)
        if record:
            rows[label] = record

    if "32-bit" not in rows:
        print("32-bit anchor missing or invalid; cannot compute retention")
        return
    anchor = rows["32-bit"]["last"]

    print(f"{'width':8s} {'Last-acc':>9s} {'Avg-acc':>8s} {'retention':>10s} "
          f"{'paper ret':>10s} {'gap':>8s}  log")
    print("-" * 92)
    for label, _case in CASES:
        if label not in rows:
            print(f"{label:8s} {'(invalid or pending)':>9s}")
            continue
        record = rows[label]
        retention = 100.0 * record["last"] / anchor
        paper_ret = PAPER[label]["retention"]
        gap = retention - paper_ret
        flag = ""
        if gap < -10.0:
            flag = "  <-- WELL BELOW PAPER"
        elif gap < -3.0:
            flag = "  <-- below paper"
        print(
            f"{label:8s} {record['last']:9.2f} {record['avg']:8.2f} "
            f"{retention:9.1f}% {paper_ret:9.1f}% {gap:+7.1f}  {record['log']}{flag}"
        )
    print("-" * 92)
    print(f"retention = Last-acc / our 32-bit Last-acc ({anchor:.2f}%)")
    print("\nAbsolute accuracies are NOT comparable to the paper: the paper reports")
    print("ACC on CIFAR-100 with 10 tasks, a from-scratch ResNet-20 and every layer")
    print("trainable, while this project uses 5 tabular tasks and a frozen ViT")
    print("backbone (5.9M of 91M parameters trainable), so low-bit truncation of the")
    print("frozen backbone cannot be recovered by training. Only retention is")
    print("informative, and even that is bounded by that structural difference.")


if __name__ == "__main__":
    main()
