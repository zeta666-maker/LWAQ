"""Size-first accuracy sweep for the 4-bit and 2-bit AQCL deployment.

"Size-first" means the frozen backbone floor equals the nominal width, so the
deployed model is ~355 / 89 / 45 / 24 MB for 32/8/4/2-bit. Everything here keeps
that size and only changes training-side or quantization-parameter choices:

  e8    baseline 8 epochs (reference, already measured)
  e16   16 epochs: QAT needs steps to adapt to a coarse grid
  scale learnable per-channel step (LSQ-style), 8 epochs

Usage:
  python tools/run_size_first.py --variants e16 --bits 4,2 --seeds 2024,2025
  python tools/run_size_first.py --variants scale --bits 2 --seeds 2024
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

RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)
SIZE_RE = re.compile(
    r"AQCL deployment: size=([\d.]+)MB, GFLOPs=([\d.]+), GBOPS=([\d.]+).*"
    r"weight_bits=(\[[^\]]*\])"
)


def build(bits: int, variant: str, seed: int, args) -> tuple[list[str], str]:
    epochs = args.epochs
    scale = False
    tracking = "frozen"
    hadamard = False
    scope = "attention"
    if variant == "e16":
        epochs = 16
    elif variant == "scale":
        scale = True
    elif variant == "e16ema":
        # Long schedule plus a live activation range: the frozen grid cannot
        # follow drifting activation statistics, which is why 16 epochs alone was
        # worse than 8.
        epochs = 16
        tracking = "ema"
    elif variant == "e8ema":
        tracking = "ema"
    elif variant == "had":
        # Hadamard-rotated quantization basis: flattens weight outliers so the
        # same bit-width has a much smaller error. No deployment size cost.
        hadamard = True
    elif variant == "had_fc2":
        hadamard = True
        scope = "attn_fc2"
    elif variant == "had_nofc1":
        hadamard = True
        scope = "no_fc1"
    elif variant == "had_e16":
        hadamard = True
        epochs = 16
    elif variant != "e8":
        raise SystemExit(f"unknown variant {variant}")

    suffix = f"{args.prefix}_fixed{bits}_{variant}_s{seed}"
    command = [
        sys.executable, "-u", str(ROOT / "train_aqcl.py"),
        "-d", "fault_csv", "-t", "5",
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32", "-e", str(epochs),
        "-jt", "0", "-je", "0", "-et", "1",
        "--eval_batch_size", "64", "--temperature", "28.0",
        "--lr", "0.005", "--lr_scale", "0.2", "--lr_scale_patterns", "patch_embed",
        "--null_eta1", "0.999", "--null_eta2", "0.999", "-tc", "2.0",
        "--eval-tool", "adapter",
        # Same evaluation protocol as the main TC-MoA matrix.
        "--eval-trained-task-router", "false",
        "--eval-task-weight", "true",
        "--eval-local-task-head", "true",
        "--prototype-confidence-weight", "0.0",
        "--seed", str(seed),
        "--save-model", "false", "--logs-dir", "logs", "-sf", suffix,
        "--aqcl-mode", "fixed",
        "--aqcl-bits", str(bits),
        "--aqcl-low-bits", str(bits),
        "--aqcl-high-bits", str(bits),
        # Size-first: the frozen backbone uses the same width as the nominal case.
        "--aqcl-backbone-bits", "-1",
        "--aqcl-fisher-batches", "8",
        "--aqcl-warmup-epochs", "1",
        "--aqcl-calibrate-batches", "4",
        "--aqcl-per-token-activation", "true",
        "--aqcl-range-tracking", tracking,
        "--aqcl-instrument", "true",
        "--aqcl-instrument-verbose", "false",
    ]
    if scale:
        command += ["--aqcl-learnable-scale", "true", "--aqcl-scale-lr", "0.001"]
    if hadamard:
        command += ["--aqcl-hadamard", "true", "--aqcl-hadamard-scope", scope]
    return command, suffix


def parse(path: Path) -> dict | None:
    """Parse final metrics, or ``None`` when the run is diverged/incomplete.

    The validity guard must live here, not only at the call sites: an interrupted
    process makes ``run_one`` return ``"ok"`` (it polls ``process.poll()``, which
    cannot distinguish "finished" from "killed by someone else"), so a log with 2
    of 5 tasks would otherwise be reported as a finished result.
    """
    if not path.exists():
        return None
    if not is_valid(path):
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    results = RESULT_RE.findall(text)
    if not results:
        return None
    sizes = SIZE_RE.findall(text)
    return {
        "log": str(path),
        "last_acc": float(results[-1][1]),
        "avg_acc": float(results[-1][2]),
        "forgetting": float(results[-1][3]),
        "size_mb": float(sizes[-1][0]) if sizes else float("nan"),
        "bits": sizes[-1][3] if sizes else "-",
    }


def is_valid(path: Path) -> bool:
    """Reject diverged or incomplete runs before they reach a results table.

    Without this guard a run that hit ``loss=nan`` and stopped after 2 of 5 tasks
    was reported as a 2-bit accuracy of 58.86% (really the task-2 accuracy of a
    collapsed model), while the healthy run of the same seed reached 42.55%.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_validity import check_run  # noqa: PLC0415

    validity = check_run(path, expected_tasks=5)
    if not validity.ok:
        print(f"[sf] {path.name}: INVALID - {validity.reason}", flush=True)
    return validity.ok


def run_one(command: list[str], log_path: Path, suffix: str, args) -> str:
    """Run one training command, guarding against a genuine hang.

    A fixed wall-clock timeout is wrong here: 2-bit runs legitimately take 33+
    minutes (``batch_time`` varies with the quantization-induced activation
    magnitudes), so the earlier 40-minute cap killed healthy runs and they were
    then misread as "stalls". The guard is therefore progress-based: a run is
    abandoned only when its own log file has not grown for ``stall_seconds``,
    with ``max_seconds`` as a generous hard cap.

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
            code = process.poll()
            if code is not None:
                if code != 0:
                    print(
                        f"[sf] {suffix}: child exited with code {code}",
                        flush=True,
                    )
                    return f"exit{code}"
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
    parser.add_argument("--variants", default="e16")
    parser.add_argument("--bits", default="4,2")
    parser.add_argument("--seeds", default="2024,2025")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--prefix", default="sf")
    parser.add_argument(
        "--stall-seconds", type=int, default=1200,
        help="Abandon a run only after its log file has not grown for this long. "
        "Progress-based, because low-bit runs vary legitimately in duration.",
    )
    parser.add_argument(
        "--max-seconds", type=int, default=10800,
        help="Hard cap per run (3 hours by default).",
    )
    parser.add_argument("--timeout", type=int, default=2400, help="Deprecated; ignored.")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bit_list = [int(b) for b in args.bits.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    work = ROOT.parent / "_aqcl_work"
    work.mkdir(parents=True, exist_ok=True)

    summary: dict[str, list[dict]] = {}
    for bits in bit_list:
        for variant in variants:
            for seed in seeds:
                command, suffix = build(bits, variant, seed, args)
                log_path = ROOT / "logs" / f"{suffix}.txt"
                key = f"fixed{bits}_{variant}"
                if log_path.exists():
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                    # A complete run needs all 5 task lines *and* no divergence.
                    if "Results of task [5]" in text and is_valid(log_path):
                        print(f"[sf] skip {suffix} (complete)", flush=True)
                        record = parse(log_path)
                        if record:
                            summary.setdefault(key, []).append(record)
                        continue
                print(f"[sf] run {suffix}", flush=True)
                outcome = run_one(command, log_path, suffix, args)
                if outcome != "ok":
                    print(f"[sf] {suffix}: ABANDONED ({outcome})", flush=True)
                    continue
                record = parse(log_path)
                if record is None:
                    print(f"[sf] {suffix}: NO VALID RESULT (diverged or incomplete)", flush=True)
                    continue
                summary.setdefault(key, []).append(record)
                print(
                    f"[sf] {suffix}: Last {record['last_acc']:.2f}% "
                    f"Avg {record['avg_acc']:.2f}% Forg {record['forgetting']:.2f}% "
                    f"size {record['size_mb']:.2f}MB",
                    flush=True,
                )

    print()
    print("| Variant | seeds | Last-acc mean | min | max | Avg-acc mean | size MB |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for key, records in summary.items():
        lasts = [r["last_acc"] for r in records]
        avgs = [r["avg_acc"] for r in records]
        sizes = [r["size_mb"] for r in records if r["size_mb"] == r["size_mb"]]
        print(
            f"| {key} | {len(records)} | {statistics.mean(lasts):.2f}% | "
            f"{min(lasts):.2f}% | {max(lasts):.2f}% | "
            f"{statistics.mean(avgs):.2f}% | "
            f"{(statistics.mean(sizes) if sizes else float('nan')):.2f} |"
        )


if __name__ == "__main__":
    main()
