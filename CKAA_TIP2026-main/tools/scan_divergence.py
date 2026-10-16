"""Scan every log for numerical divergence and incomplete task runs.

A run that reaches ``loss=nan`` or that finishes fewer than the expected number of
tasks must not enter a results table: its remaining metrics are not measurements
of the configuration. This script exists because such a run was once reported as a
valid 2-bit result (58.86%, which was really the task-2 accuracy of a run that
diverged on task 3).

Usage:
  python tools/scan_divergence.py --tasks 5
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    log_dir = ROOT / args.logs_dir
    affected: list[tuple[str, int, int, str]] = []
    total = 0
    for path in sorted(log_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        total += 1
        nan_epochs = len(re.findall(r"loss=nan", text))
        tasks = len(re.findall(r"Results of task \[", text))
        if not tasks:
            continue
        if nan_epochs or tasks < args.tasks:
            reasons = []
            if nan_epochs:
                reasons.append(f"nan x{nan_epochs}")
            if tasks < args.tasks:
                reasons.append(f"only {tasks}/{args.tasks} tasks")
            affected.append((path.name, nan_epochs, tasks, ", ".join(reasons)))

    print(f"scanned {total} logs; {len(affected)} diverged or incomplete")
    if affected and not args.quiet:
        print(f"\n{'log':46s} {'nan':>4s} {'tasks':>6s}  reason")
        for name, nan, tasks, reason in affected:
            print(f"{name:46s} {nan:4d} {tasks:6d}  {reason}")


if __name__ == "__main__":
    main()
