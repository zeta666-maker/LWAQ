"""Aggregate AQCL / CKAA training logs into a results table.

Reads every `logs/*.txt` (or the files given on the command line), extracts the
per-task accuracy list, the final Last-acc / Avg-acc / Forgetting triple, the
deployment report and the Phase-1 instrumentation summary, then prints a
Markdown table plus a JSON dump. Every number is quoted together with the log
path it came from, so no result can be reported without its evidence.

Usage:
  python tools/aqcl_summary.py --logs-dir logs --pattern "aqcl_*"
  python tools/aqcl_summary.py logs/a.txt logs/b.txt --out summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
DEPLOY_RE = re.compile(
    r"AQCL deployment: size=([\d.]+)MB, GFLOPs=([\d.]+), GBOPS=([\d.]+), "
    r"extra_memory=([\d.]+)MB, weight_bits=(\[[^\]]*\]), "
    r"activation_bits=(\[[^\]]*\])"
)
SENS_RE = re.compile(
    r"\[AQCL-INSTRUMENT\]\[rpq_sensitivity\] task=(\d+) layers=(\d+) "
    r"w_sens\[min=([\d.eE+-]+) max=([\d.eE+-]+) mean=([\d.eE+-]+) "
    r"std=([\d.eE+-]+) zeros=(\d+)\] "
    r"a_sens\[min=([\d.eE+-]+) max=([\d.eE+-]+) mean=([\d.eE+-]+) "
    r"zeros=(\d+)\] w_bits=(\{[^}]*\}) a_bits=(\{[^}]*\}) "
    r"degenerate=(\[[^\]]*\])"
)
SUBSPACE_RE = re.compile(
    r"\[AQCL-INSTRUMENT\]\[saou_subspace\] task=(\d+) layers=(\d+)"
)
SAOU_RE = re.compile(
    r"\[AQCL-INSTRUMENT\]\[saou_gradient\] task=(\d+) modulated_layers=(\d+) "
    r"steps=(\d+) cos_step_mean\[min=([\d.]+) mean=([\d.]+) max=([\d.]+)\] "
    r"gamma_step_mean\[min=([\d.]+) mean=([\d.]+)\] triggered=(\d+) skipped=(\d+)"
)
FISHER_RE = re.compile(
    r"\[AQCL-INSTRUMENT\]\[fisher_summary\] task=(\d+) targets=(\d+) "
    r"grad_None_layers=(\d+) grad_all_zero_layers=(\d+) "
    r"grad_nonzero_layers=(\d+)"
)
LOGITS_RE = re.compile(
    r"\[AQCL-INSTRUMENT\]\[logits\] task=(\d+) epoch=(\d+) "
    r"quant_enabled=(\w+) quant_absmax=([\d.nan]+) quant_saturated=([\d.nan]+) "
    r"fp32_absmax=([\d.nan]+) fp32_saturated=([\d.nan]+)"
)
KERAS_INSTR_RE = re.compile(r"\[AQCL-INSTRUMENT\]\[(rpq_sensitivity|saou_subspace|saou_gradient|deployment_bits|fisher_summary|logits)\]")


def parse_log(path: Path) -> dict:
    info: dict = {
        "log": str(path),
        "name": path.stem,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "per_task": [],
        "final": None,
        "deployment": [],
        "instrumentation": {
            "fisher": [],
            "sensitivity": [],
            "subspace": [],
            "saou": [],
            "logits": [],
        },
        "has_instrument_json": False,
    }
    instrument_json = path.with_suffix(".instrument.json")
    if instrument_json.exists():
        info["has_instrument_json"] = True
        info["instrument_json"] = str(instrument_json)

    if not path.exists():
        return info
    text = path.read_text(encoding="utf-8", errors="replace")

    for match in RESULT_RE.finditer(text):
        info["per_task"].append(
            {
                "task": int(match.group(1)),
                "last_acc": float(match.group(2)),
                "avg_acc": float(match.group(3)),
                "forgetting": float(match.group(4)),
            }
        )
    if info["per_task"]:
        info["final"] = info["per_task"][-1]

    for match in DEPLOY_RE.finditer(text):
        info["deployment"].append(
            {
                "size_mb": float(match.group(1)),
                "gflops": float(match.group(2)),
                "gbops": float(match.group(3)),
                "extra_memory_mb": float(match.group(4)),
                "weight_bits": match.group(5),
                "activation_bits": match.group(6),
            }
        )

    for match in FISHER_RE.finditer(text):
        info["instrumentation"]["fisher"].append(
            {
                "task": int(match.group(1)),
                "targets": int(match.group(2)),
                "grad_none_layers": int(match.group(3)),
                "grad_all_zero_layers": int(match.group(4)),
                "grad_nonzero_layers": int(match.group(5)),
            }
        )
    for match in SENS_RE.finditer(text):
        info["instrumentation"]["sensitivity"].append(
            {
                "task": int(match.group(1)),
                "layers": int(match.group(2)),
                "w_sens_min": float(match.group(3)),
                "w_sens_max": float(match.group(4)),
                "w_sens_mean": float(match.group(5)),
                "w_sens_std": float(match.group(6)),
                "w_sens_zeros": int(match.group(7)),
                "a_sens_min": float(match.group(8)),
                "a_sens_max": float(match.group(9)),
                "a_sens_mean": float(match.group(10)),
                "a_sens_zeros": int(match.group(11)),
                "w_bits": match.group(12),
                "a_bits": match.group(13),
                "degenerate": match.group(14),
            }
        )
    for match in SUBSPACE_RE.finditer(text):
        info["instrumentation"]["subspace"].append(
            {"task": int(match.group(1)), "layers": int(match.group(2))}
        )
    for match in SAOU_RE.finditer(text):
        info["instrumentation"]["saou"].append(
            {
                "task": int(match.group(1)),
                "modulated_layers": int(match.group(2)),
                "steps": int(match.group(3)),
                "cos_min": float(match.group(4)),
                "cos_mean": float(match.group(5)),
                "cos_max": float(match.group(6)),
                "gamma_min": float(match.group(7)),
                "gamma_mean": float(match.group(8)),
                "triggered": int(match.group(9)),
                "skipped": int(match.group(10)),
            }
        )
    for match in LOGITS_RE.finditer(text):
        info["instrumentation"]["logits"].append(
            {
                "task": int(match.group(1)),
                "epoch": int(match.group(2)),
                "quant_enabled": match.group(3) == "True",
                "quant_absmax": float(match.group(4)),
                "quant_saturated": float(match.group(5)),
                "fp32_absmax": float(match.group(6)),
                "fp32_saturated": float(match.group(7)),
            }
        )
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", help="Explicit log files.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--instrumentation-only",
        action="store_true",
        help="Only keep logs that contain Phase-1 instrumentation records.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.logs]
    if not paths:
        paths = sorted(Path(args.logs_dir).glob(args.pattern))
    records = [parse_log(p) for p in paths]
    if args.instrumentation_only:
        records = [
            r
            for r in records
            if any(r["instrumentation"][key] for key in r["instrumentation"])
        ]

    print("| Log | Last-acc | Avg-acc | Forgetting | Size MB | GFLOPs | GBOPS | w_bits |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for record in records:
        final = record["final"]
        deploy = record["deployment"][-1] if record["deployment"] else None
        print(
            "| {name} | {la} | {aa} | {fg} | {size} | {gflops} | {gbops} | {bits} |".format(
                name=record["name"],
                la="-" if not final else f"{final['last_acc']:.2f}%",
                aa="-" if not final else f"{final['avg_acc']:.2f}%",
                fg="-" if not final else f"{final['forgetting']:.2f}%",
                size="-" if not deploy else f"{deploy['size_mb']:.4f}",
                gflops="-" if not deploy else f"{deploy['gflops']:.4f}",
                gbops="-" if not deploy else f"{deploy['gbops']:.4f}",
                bits="-" if not deploy else deploy["weight_bits"],
            )
        )

    print()
    print("## Phase-1 instrumentation summary")
    for record in records:
        inst = record["instrumentation"]
        if not any(inst[key] for key in inst):
            continue
        print(f"\n### {record['name']} ({record['log']})")
        for entry in inst["fisher"]:
            print(
                f"- task {entry['task']}: targets={entry['targets']} "
                f"grad_None={entry['grad_none_layers']} "
                f"grad_all_zero={entry['grad_all_zero_layers']} "
                f"grad_nonzero={entry['grad_nonzero_layers']}"
            )
        for entry in inst["sensitivity"]:
            print(
                f"- task {entry['task']}: w_sens[{entry['w_sens_min']:.3e},"
                f"{entry['w_sens_max']:.3e}] mean={entry['w_sens_mean']:.3e} "
                f"zeros={entry['w_sens_zeros']}/{entry['layers']} "
                f"w_bits={entry['w_bits']} a_bits={entry['a_bits']} "
                f"degenerate={entry['degenerate']}"
            )
        for entry in inst["saou"]:
            print(
                f"- task {entry['task']}: modulated={entry['modulated_layers']} "
                f"cos_mean={entry['cos_mean']:.4f} "
                f"gamma_mean={entry['gamma_mean']:.4f} "
                f"triggered={entry['triggered']} skipped={entry['skipped']}"
            )
        for entry in inst["logits"]:
            print(
                f"- task {entry['task']} epoch {entry['epoch']}: "
                f"quant_enabled={entry['quant_enabled']} "
                f"quant_absmax={entry['quant_absmax']:.3f} "
                f"fp32_absmax={entry['fp32_absmax']:.3f} "
                f"quant_saturated={entry['quant_saturated']:.3f}"
            )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
