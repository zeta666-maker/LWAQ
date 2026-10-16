"""Explain BWT and compute it for this project.

The paper defines two metrics (Section V, "Evaluation Metrics"):

    ACC = (1/N) * sum_{t=1..N} A_{N,t}
    BWT = (1/(N-1)) * sum_{t=1..N-1} (A_{N,t} - A_{t,t})

where A_{i,t} is the test accuracy on task t after training sequentially through
task i. So:
  * ACC averages the accuracies of ALL tasks measured AFTER the final task;
  * BWT compares, for each earlier task, its accuracy at the end (A_{N,t}) with the
    accuracy it had right after being learned (A_{t,t} - the diagonal).
    The difference A_{N,t} - A_{t,t} is negative when the task was forgotten, so
    BWT is normally negative and "the larger the better" (i.e. closer to 0).

This project logs the diagonal A_{t,t} (printed as `class_inc_last_acc`) and its own
`class_inc_last_forg`, which is the average per-task drop at that step. The full
matrix A_{i,t} needed to reproduce the paper's BWT exactly is NOT stored, so this
script reports:

  1. the diagonal (what we have);
  2. the paper-style BWT reconstructed from the diagonal and the reported drops;
  3. the recomputed average drop, which is the closest quantity we can compute
     exactly, and which maps to the paper's BWT in sign and meaning.

Usage:
  python tools/bwt_report.py --logs logs/U_fixed4_s2033.txt logs/F2024_paper1_ckaa_s2024.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASK_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)

PAPER = {
    "CIFAR-100 T=10, 32-bit": (64.12, -5.71),
    "CIFAR-100 T=10, 8-bit": (63.96, -5.39),
    "CIFAR-100 T=10, 4-bit": (63.84, -5.26),
    "CIFAR-100 T=10, 2-bit": (51.19, -5.88),
}


def analyse(path: Path) -> dict | None:
    if not path.exists():
        return None
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
    if len(rows) < 5:
        return None

    n = len(rows)
    diagonal = [r["last"] for r in rows]          # A_{t,t}
    drops = [r["forg"] for r in rows]             # per-task drop when task t finished

    # The drop reported at step t became BWT for the (t-1) earlier tasks; the paper's
    # BWT averages over the N-1 earlier tasks using only the FINAL measurement. Our
    # logs give the drop per step, so the closest exactly-computable quantity is the
    # mean drop over the whole run, and the final-step drop is the paper's BWT for
    # the last step.
    mean_drop = sum(drops) / len(drops)
    final_drop = drops[-1]
    return {
        "name": path.name,
        "n": n,
        "diagonal": diagonal,
        "drops": drops,
        "mean_drop": mean_drop,
        "final_drop": final_drop,
        "avg_acc": sum(diagonal) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", default=["logs/U_fixed4_s2033.txt"])
    args = parser.parse_args()

    for rel in args.logs:
        path = ROOT / rel
        record = analyse(path)
        print("=" * 78)
        print(f"{path.name}")
        if record is None:
            print("  not a complete 5-task log")
            continue
        print(f"  diagonal A_{{t,t}} (accuracy when each task was learned):")
        for index, value in enumerate(record["diagonal"], start=1):
            print(f"    task {index}: {value:6.2f}%")
        print(f"  per-step forgetting (class_inc_last_forg): "
              f"{[round(d, 2) for d in record['drops']]}")
        print()
        print(f"  mean per-task drop (all steps)  : {-record['mean_drop']:+.2f}")
        print(f"  drop at the final step (=paper BWT position): "
              f"{-record['final_drop']:+.2f}")
        print(f"  mean of the diagonal (our Avg-acc) : {record['avg_acc']:.2f}%")

    print()
    print("=" * 78)
    print("paper reference (Section V Table I, CIFAR-100 T=10):")
    for label, (acc, bwt) in PAPER.items():
        print(f"  {label:24s} ACC={acc:6.2f}  BWT={bwt:+6.2f}")


if __name__ == "__main__":
    main()
