"""Build the archived four-width table from logs produced by ONE identical config.

Everything is read straight from the logs and filtered through the validity guard, so
the table cannot contain a hand-copied or a diverged number.

Usage:
  python tools/report_same_config.py --prefix U --seeds 2033,2034,2035,2036
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_validity import check_run  # noqa: E402

LAST_RE = re.compile(r"Results of task \[5\]: \[ class_inc_last_acc=([\d.]+)%")
AVG_RE = re.compile(r"Results of task \[5\]: .*class_inc_avg_acc=([\d.]+)%")
FORG_RE = re.compile(r"Results of task \[5\]: .*class_inc_last_forg=\s*([\d.]+)%")
SIZE_RE = re.compile(r"AQCL deployment: size=([\d.]+)MB")
BITS_RE = re.compile(r"weight_bits=(\[[^\]]*\])")

# label -> case suffix in the log name, and whether the case is quantized
WIDTHS = [
    ("32-bit", "fp32", False),
    ("8-bit", "fixed8", True),
    ("4-bit", "fixed4", True),
    ("2-bit", "fixed2", True),
]
PAPER_RETENTION = {"32-bit": 100.0, "8-bit": 99.8, "4-bit": 99.6, "2-bit": 79.8}


def load(prefix: str, case: str, seed: int, quantized: bool) -> dict | None:
    path = ROOT / "logs" / f"{prefix}_{case}_s{seed}.txt"
    if not path.exists():
        return {"missing": True, "seed": seed}
    validity = check_run(path, expected_tasks=5, quantized=quantized)
    text = path.read_text(encoding="utf-8", errors="replace")
    last = LAST_RE.findall(text)
    sizes = SIZE_RE.findall(text)
    bits = BITS_RE.findall(text)
    return {
        "missing": False,
        "valid": validity.ok,
        "reason": validity.reason,
        "nan": validity.nan_epochs,
        "tasks": validity.tasks_done,
        "seed": seed,
        "last": float(last[-1]) if last else None,
        "avg": float(AVG_RE.findall(text)[-1]) if AVG_RE.findall(text) else None,
        "forg": float(FORG_RE.findall(text)[-1]) if FORG_RE.findall(text) else None,
        "size": float(sizes[-1]) if sizes else None,
        "bits": bits[-1] if bits else "[32]",
        "name": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="U")
    parser.add_argument("--seeds", default="2033,2034,2035,2036")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"identical configuration, prefix={args.prefix}, seeds={seeds}")
    print()

    per_width: dict[str, list[dict]] = {}
    for label, case, quantized in WIDTHS:
        records = [load(args.prefix, case, s, quantized) for s in seeds]
        per_width[label] = records
        print(f"--- {label} ---")
        for record in records:
            if record.get("missing"):
                print(f"    seed {record['seed']}: MISSING")
            elif not record["valid"]:
                print(
                    f"    seed {record['seed']}: INVALID ({record['reason']}) "
                    f"last={record['last']} [{record['name']}]"
                )
            else:
                print(
                    f"    seed {record['seed']}: last={record['last']:6.2f} "
                    f"avg={record['avg']:6.2f} forg={record['forg']:6.2f} "
                    f"size={record['size']}MB bits={record['bits']}"
                )
        print()

    def usable(records: list[dict]) -> list[dict]:
        return [r for r in records if not r.get("missing") and r["valid"] and r["last"]]

    anchor_records = usable(per_width["32-bit"])
    anchor = statistics.mean(r["last"] for r in anchor_records) if anchor_records else None

    header = (
        f"{'width':8s} {'n':>3s} {'Last-acc':>9s} {'sd':>6s} {'min':>7s} {'max':>7s} "
        f"{'Avg-acc':>8s} {'Forg':>7s} {'size MB':>9s} {'retention':>10s} {'paper':>7s}"
    )
    print("=== ARCHIVE TABLE (valid runs only) ===")
    print(header)
    print("-" * len(header))
    for label, _case, _q in WIDTHS:
        good = usable(per_width[label])
        if not good:
            print(f"{label:8s} {0:3d} {'(no valid run)':>9s}")
            continue
        lasts = [r["last"] for r in good]
        mean = statistics.mean(lasts)
        sd = statistics.stdev(lasts) if len(lasts) > 1 else 0.0
        size = statistics.mean([r["size"] for r in good if r["size"]] or [float("nan")])
        retention = 100.0 * mean / anchor if anchor else float("nan")
        print(
            f"{label:8s} {len(good):3d} {mean:9.2f} {sd:6.2f} {min(lasts):7.2f} "
            f"{max(lasts):7.2f} {statistics.mean([r['avg'] for r in good]):8.2f} "
            f"{statistics.mean([r['forg'] for r in good]):7.2f} {size:9.2f} "
            f"{retention:9.1f}% {PAPER_RETENTION[label]:6.1f}%"
        )
    print("-" * len(header))
    if anchor:
        print(f"retention = Last-acc mean / 32-bit Last-acc mean ({anchor:.2f}%)")
    print()
    print("A width with fewer than the full seed count is reported with what is valid;")
    print("2-bit is expected to have a high invalid rate and must not be given a single")
    print("number without its spread.")


if __name__ == "__main__":
    main()
