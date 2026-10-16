"""Verify run_size_first.parse() rejects incomplete/diverged logs."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("rsf", "tools/run_size_first.py")

CASES = [
    ("logs/v2_fixed2_e8_s2033.txt", True, "complete 5-task run"),
    ("logs/v2_fixed4_e8_s2035.txt", True, "complete 5-task run"),
    ("logs/v3_fixed2_e8_s2037.txt", False, "killed at task 2 (2 task lines)"),
    ("logs/v2_fixed2_e8_s2037.txt", False, "filename collision, 2 task lines"),
    ("logs/p2_fixed2_s2024.txt", False, "diverged to nan on task 3"),
]

failures = 0
for rel, expect_result, why in CASES:
    record = runner.parse(ROOT / rel)
    ok = (record is not None) == expect_result
    if not ok:
        failures += 1
    got = "result" if record is not None else "None"
    print(f"{'PASS' if ok else 'FAIL'} {Path(rel).name:32s} -> {got:7s} ({why})")

print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
raise SystemExit(1 if failures else 0)
