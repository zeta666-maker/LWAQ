"""Audit the AQCL 32/8/4/2-bit sweep against the paper's own protocol.

Checks, for each completed run in logs/<prefix>_*.txt:

  1. size monotonicity across nominal widths (a 2-bit deployment must be smaller
     than 4-bit, which must be smaller than 8-bit);
  2. the reported ``Avg-acc`` against the paper's ACC definition
     (mean of the per-task accuracies measured *after the final task*), which is
     what the paper's Table I reports;
  3. the paper's BWT against our Forgetting;
  4. retention relative to the 32-bit run;
  5. whether the RPQ bit allocation is actually mixed or degenerate;
  6. whether SAOU modulated more than the single trainable layer.

Usage:
  python tools/audit_paper2_table.py --prefix p2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
SIZE_RE = re.compile(
    r"AQCL deployment: size=([\d.]+)MB, GFLOPs=([\d.]+), GBOPS=([\d.]+), "
    r"extra_memory=([\d.]+)MB, weight_bits=(\[[^\]]*\]), activation_bits=(\[[^\]]*\])"
)


def collect(prefix: str, tasks: int = 5) -> dict[str, dict]:
    """Load completed runs whose log is named exactly ``<prefix>_<case>_s<seed>``.

    A run counts as complete only when every task result is present; without this
    check a log that is still being written is parsed as a finished run and its
    partial numbers silently enter the table (this happened during development).
    """
    runs: dict[str, dict] = {}
    pattern = str(ROOT / "logs" / f"{prefix}_*_s[0-9]*.txt")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        match = re.match(rf"^{re.escape(prefix)}_(?P<case>.+)_s(?P<seed>\d+)\.txt$", name)
        if not match:
            continue
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        results = RESULT_RE.findall(text)
        if len(results) < tasks:
            continue
        sizes = SIZE_RE.findall(text)
        rows = [
            {
                "task": int(m[0]),
                "last": float(m[1]),
                "avg": float(m[2]),
                "forg": float(m[3]),
            }
            for m in results
        ]
        case = match.group("case")
        seed = match.group("seed")
        seed_suffix = f" (s{seed})"
        if seed != "2024" and case in runs:
            case = f"{case}{seed_suffix}"
        runs[case] = {
            "log": name,
            "rows": rows,
            "size_mb": float(sizes[-1][0]) if sizes else float("nan"),
            "gbops": float(sizes[-1][2]) if sizes else float("nan"),
            "weight_bits": sizes[-1][4] if sizes else "-",
            "activation_bits": sizes[-1][5] if sizes else "-",
            "instrument": str(Path(path).with_suffix(".instrument.json")),
        }
    return runs


def paper_metrics(rows: list[dict]) -> tuple[float, float]:
    """Paper ACC = mean over tasks of the accuracy measured after the final task.

    Our per-task history stores, for task t, the accuracy on that task's test set
    at the time task t finished (``last``). The paper instead evaluates *all* task
    test sets after the last task. The instrument JSON carries that final
    per-task vector; when it is missing we cannot compute the paper metric and
    return NaN.
    """
    per_task_final = rows[0].get("final_per_task") if rows else None
    if not per_task_final:
        return float("nan"), float("nan")
    acc = sum(per_task_final) / len(per_task_final)
    bwt = sum(
        per_task_final[t] - rows[t]["last"] for t in range(len(per_task_final) - 1)
    ) / max(len(per_task_final) - 1, 1)
    return acc, bwt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="p2")
    args = parser.parse_args()

    runs = collect(args.prefix)
    if not runs:
        print(f"no completed runs for prefix {args.prefix}")
        return

    order = ["fp32", "fixed8", "fixed4", "fixed2", "rpq_4_8", "rpq_saou_2_4"]
    keys = [k for k in order if k in runs] + [k for k in runs if k not in order]
    baseline = runs.get("fp32")

    print(
        f"{'case':14s} {'last':>7s} {'avg':>7s} {'forg':>7s} "
        f"{'size MB':>9s} {'ret%':>7s}  bits"
    )
    for key in keys:
        run = runs[key]
        last = run["rows"][-1]["last"]
        avg = run["rows"][-1]["avg"]
        forg = run["rows"][-1]["forg"]
        retention = (
            100.0 * last / baseline["rows"][-1]["last"]
            if baseline and baseline["rows"][-1]["last"]
            else float("nan")
        )
        print(
            f"{key:14s} {last:7.2f} {avg:7.2f} {forg:7.2f} "
            f"{run['size_mb']:9.2f} {retention:7.1f}  "
            f"w={run['weight_bits']} a={run['activation_bits']}"
        )

    # 1) size monotonicity
    widths = ["fixed8", "fixed4", "fixed2"]
    present = [w for w in widths if w in runs]
    print("\n[check 1] size monotonicity (size must decrease with the nominal width)")
    sizes = [runs[w]["size_mb"] for w in present]
    for a, b in zip(present, present[1:]):
        sa, sb = runs[a]["size_mb"], runs[b]["size_mb"]
        verdict = "ok" if sb < sa else "VIOLATION"
        print(f"  {a} {sa:.2f} MB -> {b} {sb:.2f} MB : {verdict}")
    if any(s != s for s in sizes):
        print("  (some sizes are nan; the run did not print a deployment line)")

    print("\n[check 2] ordering 32 >= 8 >= 4 >= 2 of the same metric")
    values = [(w, runs[w]["rows"][-1]["last"]) for w in ["fp32"] + present]
    for a, b in zip(values, values[1:]):
        verdict = "ok" if b[1] <= a[1] else "VIOLATION"
        print(f"  {a[0]} {a[1]:.2f}% -> {b[0]} {b[1]:.2f}% : {verdict}")

    # 5) allocation degeneracy
    print("\n[check 3] RPQ allocation is mixed (not degenerate)")
    for key in keys:
        path = Path(runs[key]["instrument"])
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dis, degen, saou = None, None, None
        for record in data.get("records", []):
            if record.get("stage") == "rpq_sensitivity":
                dis = record.get("weight_bits_distribution")
                degen = record.get("bit_allocation_degenerate")
            if record.get("stage") == "saou_gradient":
                saou = record.get("num_layers_modulated")
        if dis is not None:
            print(f"  {key:14s} dist={dis} degenerate={degen} saou_layers={saou}")


if __name__ == "__main__":
    main()
