"""FINAL AQCL results table, restricted to runs that pass the validity guard.

Every number here comes from a training log that
  * printed all 5 task results,
  * never reported ``loss=nan``,
  * printed one deployment line per task (or is the fp32 anchor, which prints
    none), and
  * was produced by a strictly serial run (no two training processes sharing the
    GPU).

Runs that ended early are reported separately as a failure-mode statistic, never
averaged into a result.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

TASK_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
SIZE_RE = re.compile(r"AQCL deployment: size=([\d.]+)MB.*?weight_bits=(\[[^\]]*\])")

# label -> (glob, expected weight-bit description, quantized?)
GROUPS = [
    ("32-bit fp32 (anchor)", "p2_fp32_s2024.txt", "[32]", False),
    ("8-bit uniform", "p2_fixed8_s2024.txt", "[8]", True),
    ("4-bit uniform", "v2_fixed4_e8_s20*.txt", "[4]", True),
    ("2-bit uniform", "v2_fixed2_e8_s20*.txt", "[2]", True),
    ("2-bit uniform (extra)", "v4_fixed2_e8_s2041.txt", "[2]", True),
    ("AQCL 4/8", "cm_rpq_4_8_s20*.txt", "[4, 8]", True),
    ("AQCL 2/4", "cm_rpq_saou_2_4_s20*.txt", "[2, 4]", True),
]


def runs_for(pattern: str, quantized: bool) -> tuple[list[dict], list[str]]:
    good, bad = [], []
    for path in sorted(ROOT.glob(f"logs/{pattern}")):
        validity = check_run(path, expected_tasks=5, quantized=quantized)
        if not validity.ok:
            bad.append(f"{path.name}: {validity.reason}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        tasks = TASK_RE.findall(text)
        sizes = SIZE_RE.findall(text)
        good.append(
            {
                "log": path.name,
                "last": float(tasks[-1][1]),
                "avg": float(tasks[-1][2]),
                "forg": float(tasks[-1][3]),
                "size": float(sizes[-1][0]) if sizes else None,
                "bits": sizes[-1][1] if sizes else "[32]",
            }
        )
    return good, bad


def main() -> None:
    table, excluded = [], []
    for label, pattern, bits, quantized in GROUPS:
        good, bad = runs_for(pattern, quantized)
        excluded.extend(bad)
        if not good:
            print(f"(!) {label}: no valid runs")
            continue
        lasts = [r["last"] for r in good]
        table.append(
            {
                "label": label,
                "n": len(good),
                "mean": statistics.mean(lasts),
                "sd": statistics.stdev(lasts) if len(lasts) > 1 else 0.0,
                "min": min(lasts),
                "max": max(lasts),
                "avg": statistics.mean([r["avg"] for r in good]),
                "forg": statistics.mean([r["forg"] for r in good]),
                "size": statistics.mean([r["size"] for r in good if r["size"]] or [float("nan")]),
                "bits": bits,
                "logs": [r["log"] for r in good],
            }
        )

    anchor = table[0]["mean"] if table else float("nan")
    print("=" * 108)
    print("FINAL AQCL RESULTS (valid runs only; size-first: backbone floor = nominal width)")
    print("=" * 108)
    print(
        f"{'configuration':22s} {'w_bits':>8s} {'n':>2s} {'Last-acc':>9s} {'sd':>6s} "
        f"{'min':>7s} {'max':>7s} {'Avg-acc':>8s} {'Forg':>7s} {'size MB':>8s} {'ret%':>6s}"
    )
    print("-" * 108)
    for row in table:
        ret = f"{100 * row['mean'] / anchor:5.1f}" if anchor == anchor else "-"
        print(
            f"{row['label']:22s} {row['bits']:>8s} {row['n']:2d} {row['mean']:9.2f} "
            f"{row['sd']:6.2f} {row['min']:7.2f} {row['max']:7.2f} {row['avg']:8.2f} "
            f"{row['forg']:7.2f} {row['size']:8.2f} {ret:>6s}"
        )
    print("-" * 108)
    print(f"ret% = Last-acc mean relative to the 32-bit anchor ({anchor:.2f}%)")

    print("\nper-run detail:")
    for row in table:
        print(f"  {row['label']} (n={row['n']}): " + ", ".join(row["logs"]))

    if excluded:
        print(f"\nexcluded by the validity guard ({len(excluded)}):")
        for line in excluded:
            print(f"  {line}")


if __name__ == "__main__":
    main()
