"""Correlate run failures with GPU contention (batch_time blow-up).

Hypothesis to test: the 2-bit NaN divergences and stalls are *environmental*
(several training processes sharing the 8 GB GPU), not algorithmic. If true, the
failed runs should show a much larger batch_time than the healthy runs, because a
contended GPU makes every step dramatically slower.

This matters: if contention is the cause, the fix is process discipline and the
2-bit numbers can be re-measured cleanly; if it is not, the quantization itself is
unstable and needs a code fix.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_RE = re.compile(r"batch_time=([\d.]+)")
NAN_RE = re.compile(r"loss=nan")


def profile(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    times = [float(t) for t in BATCH_RE.findall(text)]
    if not times:
        return None
    tasks = len(re.findall(r"Results of task \[", text))
    return {
        "name": path.name,
        "n": len(times),
        "median": sorted(times)[len(times) // 2],
        "max": max(times),
        "nan": len(NAN_RE.findall(text)),
        "tasks": tasks,
    }


def main() -> None:
    # Runs of interest, grouped by what we observed.
    groups = {
        "FAILED (nan divergence)": [
            "logs/p2_fixed2_s2024.txt",
            "logs/ha_fixed2_had_s2024.txt",
            "logs/ha_fixed2_had_s2026.txt",
            "logs/ha_fixed4_had_s2024.txt",
            "logs/pD_rpq_4_8.txt",
        ],
        "FAILED (stalled, no nan)": [
            "logs/nf_fixed2_e8_s2028.txt",
        ],
        "HEALTHY 2-bit": [
            "logs/tc_fixed2_s2024.txt",
            "logs/tc_fixed2_s2025.txt",
            "logs/nf_fixed2_e8_s2027.txt",
            "logs/p9_fixed2.txt",
        ],
        "HEALTHY other widths": [
            "logs/p2_fp32_s2024.txt",
            "logs/p2_fixed8_s2024.txt",
            "logs/p2_fixed4_s2024.txt",
            "logs/p2_rpq_4_8_s2024.txt",
            "logs/tc_fixed4_s2024.txt",
        ],
    }

    print(f"{'group':26s} {'log':30s} {'n':>4s} {'median':>8s} {'max':>9s} {'nan':>4s} {'tasks':>5s}")
    summary: dict[str, list[float]] = {}
    for group, paths in groups.items():
        for rel in paths:
            path = ROOT / rel
            if not path.exists():
                print(f"{group:26s} {Path(rel).name:30s}   (missing)")
                continue
            info = profile(path)
            if info is None:
                continue
            summary.setdefault(group, []).append(info["median"])
            print(
                f"{group:26s} {info['name']:30s} {info['n']:4d} "
                f"{info['median']:8.2f} {info['max']:9.2f} {info['nan']:4d} {info['tasks']:5d}"
            )

    print("\nmedian batch_time by group (seconds):")
    for group, values in summary.items():
        print(f"  {group:26s} n={len(values):2d} mean_median={sum(values) / len(values):9.2f}")


if __name__ == "__main__":
    main()
