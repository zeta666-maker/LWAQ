"""Verify the validity guard rejects the known-bad logs and keeps the good ones."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module namespace.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validity = load_module("run_validity", "tools/run_validity.py")

CASES = [
    # (log, should_be_valid, why)
    ("logs/p2_fixed2_s2024.txt", False, "diverged on task 3, only 2/5 tasks"),
    ("logs/tc_fixed2_s2024.txt", True, "healthy 2-bit baseline"),
    ("logs/p2_fixed4_s2024.txt", True, "healthy 4-bit"),
    ("logs/p2_fixed8_s2024.txt", True, "healthy 8-bit"),
    ("logs/p2_rpq_4_8_s2024.txt", True, "healthy AQCL 4/8"),
    ("logs/p2_rpq_saou_2_4_s2024.txt", True, "healthy AQCL 2/4"),
    ("logs/p2_fp32_s2024.txt", False, "no deployment line (fp32 has no quantizer)"),
    ("logs/ha_fixed2_had_s2024.txt", False, "diverged 15 epochs (rotation at 2-bit)"),
    ("logs/ha_fixed4_had_s2025.txt", True, "healthy rotated 4-bit"),
]

failures = 0
for rel, expected_valid, why in CASES:
    report = validity.check_run(ROOT / rel, expected_tasks=5)
    got = report.ok
    mark = "PASS" if got == expected_valid else "FAIL"
    if got != expected_valid:
        failures += 1
    print(
        f"{mark} {Path(rel).name:38s} valid={got!s:5s} expected={expected_valid!s:5s} "
        f"({why}) :: {report.reason}"
    )

# The metric loader must return None for invalid runs, never a low score.
metrics = validity.load_metrics(ROOT / "logs/p2_fixed2_s2024.txt", expected_tasks=5)
print(f"\nload_metrics(diverged run) -> {metrics}  (must be None)")
if metrics is not None:
    failures += 1

good = validity.load_metrics(ROOT / "logs/tc_fixed2_s2024.txt", expected_tasks=5)
print(f"load_metrics(healthy run)  -> last_acc={good['last_acc']} size={good['size_mb']}")

print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
raise SystemExit(1 if failures else 0)
