"""Diagnose why 2-bit runs end early: compare termination points and rate.

Several 2-bit runs stop after 1-2 tasks with a clean process exit and no
traceback in stderr. This script quantifies how often that happens, where the
runs stop, and whether the stopping point correlates with anything visible in the
logs (last epoch reached, activation magnitude, batch_time).

Usage: python tools/diagnose_early_exit.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

EPOCH_RE = re.compile(r"Task \[(\d+)/5\] Epoch \[(\d+)/8\]")
BATCH_RE = re.compile(r"batch_time=([\d.]+)")
INSTR_RE = re.compile(r"instrument\.json \((\d+) records\)")


def describe(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    validity = check_run(path, expected_tasks=5)
    epochs = EPOCH_RE.findall(text)
    times = [float(t) for t in BATCH_RE.findall(text)]
    records = INSTR_RE.findall(text)
    last_task = int(epochs[-1][0]) if epochs else 0
    last_epoch = int(epochs[-1][1]) if epochs else 0
    return {
        "name": path.name,
        "valid": validity.ok,
        "reason": validity.reason,
        "tasks_done": validity.tasks_done,
        "last_task": last_task,
        "last_epoch": last_epoch,
        "n_epochs": len(epochs),
        "median_bt": sorted(times)[len(times) // 2] if times else float("nan"),
        "records": int(records[-1]) if records else -1,
    }


def main() -> None:
    patterns = [
        "logs/v2_fixed2_e8_s20*.txt",
        "logs/v4_fixed2_e8_s20*.txt",
        "logs/v3_fixed2_e8_s20*.txt",
        "logs/fx_fixed2_e8_s20*.txt",
        "logs/nf_fixed2_e8_s20*.txt",
    ]
    seen: set[str] = set()
    rows = []
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if path.name in seen:
                continue
            seen.add(path.name)
            rows.append(describe(path))

    print(f"{'log':30s} {'valid':>5s} {'tasks':>5s} {'lastT':>5s} {'lastE':>5s} "
          f"{'epochs':>6s} {'med_bt':>7s} {'recs':>4s}  reason")
    for r in rows:
        print(
            f"{r['name']:30s} {str(r['valid']):>5s} {r['tasks_done']:5d} "
            f"{r['last_task']:5d} {r['last_epoch']:5d} {r['n_epochs']:6d} "
            f"{r['median_bt']:7.2f} {r['records']:4d}  {r['reason'][:40]}"
        )

    valid = [r for r in rows if r["valid"]]
    bad = [r for r in rows if not r["valid"]]
    print(
        f"\n2-bit runs: {len(valid)} valid, {len(bad)} ended early "
        f"({100 * len(bad) / max(len(rows), 1):.0f}% early-exit rate)"
    )
    if bad:
        print("early exits stop at (task, epoch):")
        for r in bad:
            print(f"  {r['name']:30s} task {r['last_task']} epoch {r['last_epoch']} "
                  f"({r['n_epochs']} epochs logged, {r['records']} instrument records)")


if __name__ == "__main__":
    main()
