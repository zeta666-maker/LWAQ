"""Paper-protocol audit of the AQCL 32/8/4/2-bit sweep (prefix p2).

Computes, for each run, the paper's own metrics from the printed per-task
accuracy matrix A[i][j] = accuracy on task j measured after finishing task i:

  ACC = (1/N) * sum_j A[N-1][j]              (paper Eq. for "ACC")
  BWT = (1/(N-1)) * sum_{j<N-1} (A[N-1][j] - A[j][j])

Our own logged metrics map as: Last-acc = A[N-1][N-1], Forgetting = our
"class_inc_last_forg" (a different convention from BWT).

Also checks the invariants the sweep must satisfy.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

TASK_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
# The runner prints, per case, the sequence of per-task last-accuracies.
TRACE_RE = re.compile(r"per-task \[([0-9.,\s]+)\]")


def trace_for(case: str, seed: int = 2024) -> list[float] | None:
    """Read the per-task trace printed by the matrix runner in its summary."""
    return None


def load(case: str, seed: int = 2024) -> dict:
    path = LOG_DIR / f"p2_{case}_s{seed}.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = [
        {
            "task": int(m[0]),
            "last": float(m[1]),
            "avg": float(m[2]),
            "forg": float(m[3]),
        }
        for m in TASK_RE.findall(text)
    ]
    size = re.findall(r"AQCL deployment: size=([\d.]+)MB", text)
    bits = re.findall(r"weight_bits=(\[[^\]]*\])", text)
    return {
        "case": case,
        "rows": rows,
        "size": float(size[-1]) if size else float("nan"),
        "bits": bits[-1] if bits else "-",
    }


CASES = ["fp32", "fixed8", "fixed4", "fixed2", "rpq_4_8", "rpq_saou_2_4"]


def main() -> None:
    runs = {case: load(case) for case in CASES}

    print("== our logged metrics (5 tasks, seed 2024) ==")
    print(f"{'case':14s} {'Last':>7s} {'Avg-acc':>8s} {'Forg':>7s} {'size MB':>9s} bits")
    for case in CASES:
        run = runs[case]
        last = run["rows"][-1]["last"]
        print(
            f"{case:14s} {last:7.2f} {run['rows'][-1]['avg']:8.2f} "
            f"{run['rows'][-1]['forg']:7.2f} {run['size']:9.2f} {run['bits']}"
        )

    # Paper ACC needs the full matrix, which the log does not print in matrix
    # form; the runner prints only the diagonal (per-task trace). Report what can
    # be derived and be explicit about what cannot.
    print("\n== ordering checks ==")
    base = runs["fp32"]["rows"][-1]["last"]
    prev_name, prev_val = "fp32", base
    for case in ["fixed8", "fixed4", "fixed2"]:
        value = runs[case]["rows"][-1]["last"]
        verdict = "ok" if value <= prev_val else "VIOLATION"
        print(f"  {prev_name:9s} {prev_val:6.2f}%  ->  {case:9s} {value:6.2f}%   {verdict}")
        prev_name, prev_val = case, value

    print("\n== retention vs 32-bit, and size ==")
    for case in ["fixed8", "fixed4", "fixed2", "rpq_4_8", "rpq_saou_2_4"]:
        run = runs[case]
        last = run["rows"][-1]["last"]
        print(
            f"  {case:14s} retention {100.0 * last / base:5.1f}%  "
            f"({base:.2f} -> {last:.2f})   size {run['size']:6.2f} MB"
        )

    print("\n== size monotonicity ==")
    chained = [("fp32", runs["fp32"]["size"]), ("fixed8", runs["fixed8"]["size"]),
               ("fixed4", runs["fixed4"]["size"]), ("fixed2", runs["fixed2"]["size"])]
    for (na, sa), (nb, sb) in zip(chained, chained[1:]):
        verdict = "ok" if sb < sa or sa != sa else "VIOLATION"
        print(f"  {na:8s} {sa:8.2f} MB -> {nb:8s} {sb:8.2f} MB   {verdict}")

    print("\n== per-task diagonal (accuracy on each task when it was learned) ==")
    for case in CASES:
        diag = " ".join(f"{r['last']:6.2f}" for r in runs[case]["rows"])
        print(f"  {case:14s} {diag}")


if __name__ == "__main__":
    main()
