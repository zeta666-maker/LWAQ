"""Final AQCL comparison table, built only from runs that pass the validity guard.

Sources (all strictly serial, no overlapping training processes):
  logs/p2_*_s2024.txt       32/8/4/2-bit sweep (seed 2024)
  logs/v2_fixed2_e8_s20*.txt   uniform 2-bit, 5 seeds
  logs/v2_fixed4_e8_s20*.txt   uniform 4-bit, 3 seeds
  logs/cm_rpq_saou_2_4_s20*.txt  AQCL 2/4, 3 seeds
  logs/cm_rpq_4_8_s20*.txt       AQCL 4/8, 3 seeds

Every row is filtered through tools/run_validity.py: a run with a diverged
(``loss=nan``) or incomplete log is excluded, never averaged in.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
SIZE_RE = re.compile(r"AQCL deployment: size=([\d.]+)MB.*?weight_bits=(\[[^\]]*\])")


def load(pattern: str) -> list[dict]:
    runs = []
    for path in sorted(ROOT.glob(pattern)):
        validity = check_run(path, expected_tasks=5)
        if not validity.ok:
            print(f"  excluded {path.name}: {validity.reason}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        results = RESULT_RE.findall(text)
        sizes = SIZE_RE.findall(text)
        runs.append(
            {
                "log": path.name,
                "last": float(results[-1][1]),
                "avg": float(results[-1][2]),
                "forg": float(results[-1][3]),
                "size": float(sizes[-1][0]) if sizes else None,
                "bits": sizes[-1][1] if sizes else "-",
            }
        )
    return runs


GROUPS = [
    ("32-bit (fp32)", "logs/p2_fp32_s2024.txt"),
    ("8-bit uniform", "logs/p2_fixed8_s2024.txt"),
    ("4-bit uniform", "logs/v2_fixed4_e8_s20*.txt"),
    ("2-bit uniform", "logs/v2_fixed2_e8_s20*.txt"),
    ("AQCL 4/8 (RPQ+SAOU)", "logs/cm_rpq_4_8_s20*.txt"),
    ("AQCL 2/4 (RPQ+SAOU)", "logs/cm_rpq_saou_2_4_s20*.txt"),
]


def main() -> None:
    print("=" * 100)
    print("FINAL AQCL TABLE (valid runs only, seed 2024 for the fp32/8-bit anchors)")
    print("=" * 100)
    rows = []
    for label, pattern in GROUPS:
        # Allow an exact path or a glob.
        if "*" in pattern:
            runs = load(pattern)
        else:
            runs = load(pattern) if Path(ROOT / pattern).exists() else []
        if not runs:
            print(f"{label:24s} no valid runs")
            continue
        lasts = [r["last"] for r in runs]
        avgs = [r["avg"] for r in runs]
        sizes = [r["size"] for r in runs if r["size"]]
        rows.append(
            {
                "label": label,
                "n": len(runs),
                "mean": statistics.mean(lasts),
                "sd": statistics.stdev(lasts) if len(lasts) > 1 else 0.0,
                "min": min(lasts),
                "max": max(lasts),
                "avg": statistics.mean(avgs),
                "size": statistics.mean(sizes) if sizes else float("nan"),
                "bits": runs[0]["bits"],
            }
        )
        print(f"\n{label}  (n={len(runs)})")
        for r in runs:
            print(
                f"    {r['log']:34s} last={r['last']:6.2f} avg={r['avg']:6.2f} "
                f"forg={r['forg']:6.2f} size={r['size']}"
            )

    base = next((r for r in rows if r["label"].startswith("32-bit")), None)
    print("\n" + "=" * 100)
    print(
        f"{'configuration':24s} {'n':>2s} {'Last-acc':>9s} {'sd':>6s} {'min':>7s} "
        f"{'max':>7s} {'Avg-acc':>8s} {'size MB':>8s} {'retention':>10s}"
    )
    print("-" * 100)
    for r in rows:
        retention = f"{100 * r['mean'] / base['mean']:.1f}%" if base else "-"
        print(
            f"{r['label']:24s} {r['n']:2d} {r['mean']:9.2f} {r['sd']:6.2f} "
            f"{r['min']:7.2f} {r['max']:7.2f} {r['avg']:8.2f} {r['size']:8.2f} "
            f"{retention:>10s}"
        )
    print("-" * 100)
    print("retention = Last-acc mean / 32-bit Last-acc mean")


if __name__ == "__main__":
    main()
