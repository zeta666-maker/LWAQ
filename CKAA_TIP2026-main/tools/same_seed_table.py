"""Same-seed, same-batch comparison of all four widths, with validity.

This is the only fair horizontal comparison: identical seed, identical batch of
runs, identical protocol. It shows exactly which rows can be used as a number and
which cannot.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

LAST_RE = re.compile(r"Results of task \[5\]: \[ class_inc_last_acc=([\d.]+)%")
AVG_RE = re.compile(r"Results of task \[5\]: .*class_inc_avg_acc=([\d.]+)%")
FORG_RE = re.compile(r"Results of task \[5\]: .*class_inc_last_forg=\s*([\d.]+)%")

ROWS = [
    ("CKAA (paper 1)", "F2024_paper1_ckaa_s2024.txt", False),
    ("AQCL 32-bit", "F2024_fp32_s2024.txt", False),
    ("AQCL 8-bit", "F2024_fixed8_s2024.txt", True),
    ("AQCL 4-bit", "F2024_fixed4_s2024.txt", True),
    ("AQCL 2-bit", "F2024_fixed2_s2024.txt", True),
]


def main() -> None:
    header = (
        f"{'configuration':16s} {'valid':>5s} {'nan':>3s} {'tasks':>5s} "
        f"{'Last-acc':>9s} {'Avg-acc':>8s} {'Forg':>7s}"
    )
    print(header)
    print("-" * len(header))
    for label, name, quantized in ROWS:
        path = ROOT / "logs" / name
        validity = check_run(path, expected_tasks=5, quantized=quantized)
        text = path.read_text(encoding="utf-8", errors="replace")
        last = LAST_RE.findall(text)
        avg = AVG_RE.findall(text)
        forg = FORG_RE.findall(text)
        usable = validity.ok
        print(
            f"{label:16s} {str(usable):>5s} {validity.nan_epochs:3d} "
            f"{validity.tasks_done:5d} {last[-1] if last else '-':>9s} "
            f"{avg[-1] if avg else '-':>8s} {forg[-1] if forg else '-':>7s}"
        )
    print()
    print("A row is usable as 'the accuracy of this configuration' only if valid=True.")
    print("The 2-bit row of this batch is INVALID, so 2-bit has no single number here.")


if __name__ == "__main__":
    main()
