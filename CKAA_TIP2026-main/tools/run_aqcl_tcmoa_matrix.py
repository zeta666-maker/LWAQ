"""Full AQCL comparison matrix under one fixed, low-variance protocol.

Every case runs `train_aqcl.py` with the same CKAA/AQCL configuration and the
same evaluation protocol; only the quantization setting differs. The evaluation
protocol is the paper's TC-MoA confidence-weighted adapter mixture
(`--eval-trained-task-router false --eval-task-weight true`), which is
deterministic and therefore gives comparable numbers across cases, unlike the
repository's learned RandomForest router (see AQCL_RESULTS.md section 5).

Cases:
  fp32            no quantization
  fixed8/4/2      uniform b-bit for every layer (with the backbone floor)
  rpq_4_8         RPQ Eq.13, b_L=4, b_H=8
  rpq_saou_2_4    RPQ + SAOU, b_L=2, b_H=4

Usage:
  python tools/run_aqcl_tcmoa_matrix.py --seeds 2024,2025
  python tools/run_aqcl_tcmoa_matrix.py --seeds 2024,2025 --epochs 8 --cases fp32,fixed8
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES: dict[str, tuple[str, str, int, int, int]] = {
    "fp32": ("false", "fixed", 32, 32, 32),
    "fixed8": ("true", "fixed", 8, 8, 8),
    "fixed4": ("true", "fixed", 4, 4, 4),
    "fixed2": ("true", "fixed", 2, 2, 2),
    "rpq_4_8": ("true", "rpq", 4, 4, 8),
    "rpq_saou_2_4": ("true", "rpq_saou", 2, 2, 4),
}

# Per-case overrides applied after the base command is built.
#   backbone-bits    precision floor for the frozen backbone
#   act-low/high     decoupled activation widths (paper Eq.14/15 are separate,
#                    so weights and activations may use different widths)
#   act-bits         decoupled activation width in fixed mode
#   learnable-scale  LSQ-style learnable step multiplier (size-neutral)
#   extra            raw extra CLI flags
CASE_OVERRIDES: dict[str, dict] = {
    "rpq_saou_2_4_bb4": {"backbone-bits": 4, "act-low": 2, "act-high": 4},
    "rpq_saou_2_4_bb8": {"backbone-bits": 8, "act-low": 2, "act-high": 4},
    "rpq_saou_2_4_act4": {"backbone-bits": 4, "act-low": 4, "act-high": 4},
    "rpq_saou_2_4_act48": {"backbone-bits": 4, "act-low": 4, "act-high": 8},
    "rpq_saou_2_4_bb8_act48": {"backbone-bits": 8, "act-low": 4, "act-high": 8},
    "rpq_4_8_act48": {"backbone-bits": -1, "act-low": 4, "act-high": 8},
    "rpq_4_8_bb8": {"backbone-bits": 8, "act-low": 4, "act-high": 8},
    # Size-first improvements for the 4-bit and 2-bit deployments: the backbone
    # floor stays equal to b_H (=-1) so the deployed model size is unchanged.
    "fixed4_scale": {"backbone-bits": -1, "learnable-scale": 1},
    "fixed4_e12": {"backbone-bits": -1, "extra": ["--epochs-marker"]},
    "fixed2_scale": {"backbone-bits": -1, "learnable-scale": 1},
    "fixed2_e12": {"backbone-bits": -1},
    "fixed2_scale_e12": {"backbone-bits": -1, "learnable-scale": 1, "extra": ["--epochs-marker"]},
}
for _name, _spec in CASE_OVERRIDES.items():
    if _name.startswith("rpq_saou_2_4"):
        _base = "rpq_saou_2_4"
    elif _name.startswith("rpq_4_8"):
        _base = "rpq_4_8"
    elif _name.startswith("fixed4"):
        _base = "fixed4"
    else:
        _base = "fixed2"
    CASES[_name] = CASES[_base]

DEFAULT_CASES = (
    "fp32",
    "fixed8",
    "fixed4",
    "fixed2",
    "rpq_4_8",
    "rpq_saou_2_4",
)

RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
SIZE_RE = re.compile(
    r"AQCL deployment: size=([\d.]+)MB, GFLOPs=([\d.]+), GBOPS=([\d.]+).*"
    r"weight_bits=(\[[^\]]*\])"
)


def build(seed: int, case: str, args) -> tuple[list[str], str]:
    enable, mode, bits, low, high = CASES[case]
    suffix = f"{args.prefix}_{case}_s{seed}"
    command = [
        sys.executable, "-u", str(ROOT / "train_aqcl.py"),
        "-d", "fault_csv",
        "-t", str(args.tasks),
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32",
        "-e", str(args.epochs),
        "-jt", "0", "-je", "0", "-et", "1",
        "--eval_batch_size", "64",
        "--temperature", "28.0",
        "--lr", "0.005",
        "--lr_scale", "0.2",
        "--lr_scale_patterns", "patch_embed",
        "--null_eta1", "0.999",
        "--null_eta2", "0.999",
        "-tc", "2.0",
        "--eval-tool", "adapter",
        # Paper protocol: confidence-weighted TC-MoA, no learned router.
        "--eval-trained-task-router", "false",
        "--eval-task-weight", "true",
        "--eval-local-task-head", "true",
        "--prototype-confidence-weight", "0.0",
        "--seed", str(seed),
        "--save-model", "false",
        "--logs-dir", "logs",
        "-sf", suffix,
        "--aqcl-enable", enable,
        "--aqcl-mode", mode,
        "--aqcl-bits", str(bits),
        "--aqcl-low-bits", str(low),
        "--aqcl-high-bits", str(high),
        "--aqcl-backbone-bits", str(args.backbone_bits),
        "--aqcl-fisher-batches", "8",
        "--aqcl-warmup-epochs", "1",
        "--aqcl-calibrate-batches", "4",
        "--aqcl-per-token-activation", "true",
        "--aqcl-instrument", "true",
        "--aqcl-instrument-verbose", "false",
    ]
    override = CASE_OVERRIDES.get(case)
    if override:
        if "backbone-bits" in override:
            index = command.index("--aqcl-backbone-bits") + 1
            command[index] = str(override["backbone-bits"])
        if "act-low" in override:
            command += ["--aqcl-act-low-bits", str(override["act-low"])]
        if "act-high" in override:
            command += ["--aqcl-act-high-bits", str(override["act-high"])]
        if "act-bits" in override:
            command += ["--aqcl-act-bits", str(override["act-bits"])]
        if override.get("learnable-scale"):
            command += ["--aqcl-learnable-scale", "true"]
        for flag in override.get("extra", []):
            if flag == "--epochs-marker":
                # Bump only the epoch count of this case.
                idx = command.index("-e") + 1
                command[idx] = str(args.epochs + 4)
            else:
                command.append(flag)
    return command, suffix


def parse(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_validity import check_run  # noqa: PLC0415

    # The fp32 anchor has no quantizer and prints no deployment line, so the guard
    # must be told not to require one.
    text_probe = log_path.read_text(encoding="utf-8", errors="replace")
    is_quantized = "AQCL deployment" in text_probe
    validity = check_run(log_path, expected_tasks=5, quantized=is_quantized)
    if not validity.ok:
        print(f"[tc] {log_path.name}: INVALID - {validity.reason}", flush=True)
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    results = RESULT_RE.findall(text)
    if not results:
        return None
    sizes = SIZE_RE.findall(text)
    last_acc, avg_acc, forg = (float(v) for v in results[-1][1:])
    return {
        "log": str(log_path),
        "last_acc": last_acc,
        "avg_acc": avg_acc,
        "forgetting": forg,
        "size_mb": float(sizes[-1][0]) if sizes else float("nan"),
        "gbops": float(sizes[-1][2]) if sizes else float("nan"),
        "bits": sizes[-1][3] if sizes else "-",
        "per_task": [float(r[1]) for r in results],
    }


def run_with_progress_guard(
    command: list[str], log_path: Path, suffix: str, args
) -> str:
    """Run one case, abandoning it only when it stops making progress.

    A fixed wall-clock timeout is unsafe here: low-bit runs legitimately take
    30-40+ minutes (``batch_time`` grows with the quantization-induced activation
    magnitudes), so the old 2400 s cap killed healthy runs (e.g.
    ``cm_rpq_4_8_s2033``) which were then misread as failures.

    Returns ``"ok"``, ``"stalled"`` or ``"max"``.
    """
    work = ROOT.parent / "_aqcl_work"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with open(work / f"{suffix}.out", "wb") as out, open(
        work / f"{suffix}.err", "wb"
    ) as err:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=out, stderr=err)
        start = time.time()
        last_size, last_change = -1, time.time()
        while True:
            if process.poll() is not None:
                return "ok"
            now = time.time()
            size = log_path.stat().st_size if log_path.exists() else 0
            if size != last_size:
                last_size, last_change = size, now
            if now - last_change > args.stall_seconds:
                process.kill()
                process.wait()
                return "stalled"
            if now - start > args.max_seconds:
                process.kill()
                process.wait()
                return "max"
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--seeds", default="2024,2025")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--prefix", default="tc")
    parser.add_argument("--backbone-bits", type=int, default=-1)
    parser.add_argument(
        "--stall-seconds", type=int, default=1500,
        help="Abandon a case only after its log has not grown for this long.",
    )
    parser.add_argument(
        "--max-seconds", type=int, default=10800, help="Hard cap per case."
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Deprecated; ignored.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        raise SystemExit(f"Unknown cases {unknown}; known {sorted(CASES)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    work = ROOT.parent / "_aqcl_work"
    work.mkdir(parents=True, exist_ok=True)

    summary: dict[str, list[dict]] = {case: [] for case in cases}
    for case in cases:
        for seed in seeds:
            command, suffix = build(seed, case, args)
            log_path = ROOT / "logs" / f"{suffix}.txt"
            if args.dry_run:
                print(f"[tc] would run {suffix}")
                continue
            if not args.force and log_path.exists():
                text = log_path.read_text(encoding="utf-8", errors="replace")
                if f"Results of task [{args.tasks}]" in text:
                    print(f"[tc] skip {suffix} (complete)", flush=True)
                    record = parse(log_path)
                    if record:
                        summary[case].append(record)
                    continue
            print(f"[tc] run {suffix}", flush=True)
            outcome = run_with_progress_guard(command, log_path, suffix, args)
            if outcome != "ok":
                print(f"[tc] {suffix} ABANDONED ({outcome})", flush=True)
                continue
            record = parse(log_path)
            if record is None:
                print(f"[tc] {suffix}: no result line", flush=True)
                tail = (work / f"{suffix}.err")
                if tail.exists() and tail.stat().st_size:
                    print(tail.read_text(encoding="utf-8", errors="replace")[-1500:])
                continue
            summary[case].append(record)
            print(
                f"[tc] {suffix}: Last {record['last_acc']:.2f}% "
                f"Avg {record['avg_acc']:.2f}% Forg {record['forgetting']:.2f}% "
                f"size {record['size_mb']:.2f}MB bits {record['bits']}",
                flush=True,
            )

    print()
    print("## TC-MoA protocol summary")
    print()
    print("| Case | seeds | Last-acc mean | min | max | Avg-acc mean | size MB | bits |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for case in cases:
        records = summary[case]
        if not records:
            print(f"| {case} | - | - | - | - | - | - | - |")
            continue
        lasts = [r["last_acc"] for r in records]
        avgs = [r["avg_acc"] for r in records]
        sizes = [r["size_mb"] for r in records if r["size_mb"] == r["size_mb"]]
        print(
            f"| {case} | {len(records)} | {statistics.mean(lasts):.2f}% | "
            f"{min(lasts):.2f}% | {max(lasts):.2f}% | "
            f"{statistics.mean(avgs):.2f}% | "
            f"{(statistics.mean(sizes) if sizes else float('nan')):.2f} | "
            f"{records[-1]['bits']} |"
        )
    for case in cases:
        for record in summary[case]:
            print(f"  {case} {record['log']}: per-task {record['per_task']}")


if __name__ == "__main__":
    main()
