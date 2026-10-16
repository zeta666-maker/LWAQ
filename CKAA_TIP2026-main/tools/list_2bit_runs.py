"""List every 2-bit run with its validity and accuracy, to answer plainly:
"what IS the 2-bit number?"

Prints one line per run plus the summary statistics over valid runs.
"""

from __future__ import annotations

import glob
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

FINAL_RE = re.compile(
    r"Results of task \[5\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)%"
)


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "logs" / "*fixed2*.txt"))):
        name = os.path.basename(path)
        if any(tag in name for tag in ("crash", "faulthandler", "instrument")):
            continue
        validity = check_run(path, expected_tasks=5)
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        match = FINAL_RE.findall(text)
        rows.append((name, validity, match[-1] if match else None))

    header = f"{'log':34s} {'valid':>5s} {'tasks':>5s} {'nan':>3s} {'last':>7s} {'avg':>7s} {'forg':>7s}"
    print(header)
    print("-" * len(header))
    for name, validity, metrics in rows:
        last, avg, forg = metrics if metrics else ("-", "-", "-")
        print(
            f"{name:34s} {str(validity.ok):>5s} {validity.tasks_done:5d} "
            f"{validity.nan_epochs:3d} {last:>7s} {avg:>7s} {forg:>7s}"
        )

    valid = [(n, float(m[0]), float(m[1]), float(m[2])) for n, v, m in rows if v.ok and m]
    print()
    print(f"valid runs: {len(valid)} of {len(rows)}")
    if valid:
        lasts = [v[1] for v in valid]
        print(
            f"  Last-acc  mean={statistics.mean(lasts):.2f}  "
            f"sd={statistics.stdev(lasts):.2f}  min={min(lasts):.2f}  max={max(lasts):.2f}"
        )
        best = max(valid, key=lambda r: r[1])
        worst = min(valid, key=lambda r: r[1])
        print(f"  best  : {best[0]} -> last={best[1]:.2f} avg={best[2]:.2f} forg={best[3]:.2f}")
        print(f"  worst : {worst[0]} -> last={worst[1]:.2f} avg={worst[2]:.2f} forg={worst[3]:.2f}")


if __name__ == "__main__":
    main()
