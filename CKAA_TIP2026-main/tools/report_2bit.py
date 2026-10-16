"""Report the validity and final accuracy of a set of 2-bit logs."""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

LAST_RE = re.compile(r"Results of task \[5\]: \[ class_inc_last_acc=([\d.]+)%")
AVG_RE = re.compile(r"Results of task \[5\]: .*class_inc_avg_acc=([\d.]+)%")

pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/RF_fixed2*.txt"
valid_values = []
print(f"{'log':34s} {'valid':>5s} {'tasks':>5s} {'nan':>4s} {'last':>8s} {'avg':>8s}")
for path in sorted(glob.glob(str(ROOT / pattern))):
    name = os.path.basename(path)
    if "crash" in name or "faulthandler" in name:
        continue
    validity = check_run(path, expected_tasks=5)
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    last = LAST_RE.findall(text)
    avg = AVG_RE.findall(text)
    if validity.ok and last:
        valid_values.append(float(last[-1]))
    print(
        f"{name:34s} {str(validity.ok):>5s} {validity.tasks_done:5d} "
        f"{validity.nan_epochs:4d} {last[-1] if last else '-':>8s} "
        f"{avg[-1] if avg else '-':>8s}"
    )

if valid_values:
    import statistics

    print(
        f"\nvalid 2-bit runs: n={len(valid_values)} "
        f"mean={statistics.mean(valid_values):.2f} "
        f"min={min(valid_values):.2f} max={max(valid_values):.2f}"
    )
else:
    print("\nno valid 2-bit run in this set")
