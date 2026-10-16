"""Strictly same-configuration comparison: all widths on identical seeds.

Every row must come from
  * the same prefix (therefore the same CLI configuration and code path),
  * the same seed set,
  * a log that passes the validity guard.

Anything not meeting all three is reported as a gap to be filled, never guessed.
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

# prefix -> {width label: case name suffix used by that runner}
PREFIXES = {
    "F2024": {"32-bit": "fp32", "8-bit": "fixed8", "4-bit": "fixed4", "2-bit": "fixed2"},
    "F2033": {"32-bit": "fp32", "8-bit": "fixed8", "4-bit": "fixed4", "2-bit": "fixed2"},
    "F2034": {"32-bit": "fp32", "8-bit": "fixed8", "4-bit": "fixed4", "2-bit": "fixed2"},
    "F2035": {"32-bit": "fp32", "8-bit": "fixed8", "4-bit": "fixed4", "2-bit": "fixed2"},
    "F2036": {"32-bit": "fp32", "8-bit": "fixed8", "4-bit": "fixed4", "2-bit": "fixed2"},
}
SEEDS = [2033, 2034, 2035, 2036]


def read(prefix: str, case: str, seed: int) -> dict | None:
    path = ROOT / "logs" / f"{prefix}_{case}_s{seed}.txt"
    if not path.exists():
        return None
    quantized = case != "fp32"
    validity = check_run(path, expected_tasks=5, quantized=quantized)
    text = path.read_text(encoding="utf-8", errors="replace")
    last = LAST_RE.findall(text)
    return {
        "valid": validity.ok,
        "reason": validity.reason,
        "nan": validity.nan_epochs,
        "last": float(last[-1]) if last else None,
        "avg": float(AVG_RE.findall(text)[-1]) if AVG_RE.findall(text) else None,
        "forg": float(FORG_RE.findall(text)[-1]) if FORG_RE.findall(text) else None,
        "name": path.name,
    }


def main() -> None:
    for prefix, cases in PREFIXES.items():
        print(f"\n=== prefix {prefix} (identical CLI configuration) ===")
        header = f"{'width':8s}" + "".join(f"{('s' + str(s)):>12s}" for s in SEEDS)
        print(header)
        print("-" * len(header))
        for label, case in cases.items():
            cells = []
            for seed in SEEDS:
                record = read(prefix, case, seed)
                if record is None:
                    cells.append("missing")
                elif not record["valid"]:
                    cells.append(f"INV(nan={record['nan']})")
                else:
                    cells.append(f"{record['last']:.2f}")
            print(f"{label:8s}" + "".join(f"{c:>12s}" for c in cells))

    print("\n=== which (prefix, width) have ALL four seeds valid? ===")
    for prefix, cases in PREFIXES.items():
        for label, case in cases.items():
            records = [read(prefix, case, s) for s in SEEDS]
            good = [r for r in records if r and r["valid"]]
            if len(good) == len(SEEDS):
                values = [r["last"] for r in good]
                mean = sum(values) / len(values)
                print(f"  {prefix}/{label}: COMPLETE  mean={mean:.2f}  values={values}")
            else:
                print(f"  {prefix}/{label}: incomplete ({len(good)}/{len(SEEDS)} valid)")


if __name__ == "__main__":
    main()
