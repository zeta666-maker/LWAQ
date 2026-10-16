"""Run everything the final report needs, strictly serially.

Order:
  1. AQCL 32-bit (fp32 anchor), 8-bit, 4-bit, 2-bit  -- via run_aqcl_tcmoa_matrix
  2. CKAA paper-1 baseline (12 epochs)                -- via train_eval.py (unmodified)

One seed per invocation so the chain can be resumed; each run is guarded so a
diverged or incomplete log can never be reported as a result.

Usage: python tools/run_final_sweep.py --seed 2024
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
WORK = ROOT.parent / "_aqcl_work"
TASK_RE = re.compile(r"Results of task \[")
RESULT_RE = re.compile(
    r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
    r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]"
)


def wait_for(log_path: Path, stall: int, cap: int) -> str:
    start = time.time()
    last_size, last_change = -1, time.time()
    while True:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if len(TASK_RE.findall(text)) >= 5:
                return "complete"
        now = time.time()
        size = log_path.stat().st_size if log_path.exists() else 0
        if size != last_size:
            last_size, last_change = size, now
        if now - last_change > stall:
            return "stalled"
        if now - start > cap:
            return "max"
        time.sleep(10)


def run(command: list[str], log_path: Path, tag: str, stall: int, cap: int) -> str:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with open(WORK / f"{tag}.out", "wb") as out, open(WORK / f"{tag}.err", "wb") as err:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=out, stderr=err)
        outcome = wait_for(log_path, stall, cap)
        if process.poll() is None and outcome != "complete":
            process.kill()
        process.wait()
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--stall", type=int, default=1800)
    parser.add_argument("--cap", type=int, default=10800)
    args = parser.parse_args()

    prefix = f"F{args.seed}"
    rows: list[tuple[str, float, float, float, str]] = []

    # 1) AQCL widths, one case per invocation so each has its own progress guard.
    for case in ("fp32", "fixed8", "fixed4", "fixed2"):
        log_path = ROOT / "logs" / f"{prefix}_{case}_s{args.seed}.txt"
        if wait_for(log_path, 5, 5) == "complete":
            print(f"[final] {case}: already complete", flush=True)
        else:
            outcome = run(
                [
                    sys.executable, "-u", str(ROOT / "tools" / "run_aqcl_tcmoa_matrix.py"),
                    "--cases", case, "--seeds", str(args.seed),
                    "--epochs", str(args.epochs), "--prefix", prefix,
                    "--stall-seconds", str(args.stall), "--max-seconds", str(args.cap),
                ],
                log_path, f"{prefix}_{case}", args.stall, args.cap,
            )
            print(f"[final] {case}: {outcome}", flush=True)

    # 2) CKAA paper-1 baseline with the unmodified entry point.
    ckaa_suffix = f"{prefix}_paper1_ckaa_s{args.seed}"
    ckaa_log = ROOT / "logs" / f"{ckaa_suffix}.txt"
    if wait_for(ckaa_log, 5, 5) != "complete":
        outcome = run(
            [
                sys.executable, "-u", str(ROOT / "train_eval.py"),
                "-d", "fault_csv", "-t", "5",
                "-m", "vit_base_patch16_224.augreg_in21k",
                "-b", "32", "-e", "12", "-jt", "0", "-je", "0", "-et", "1",
                "--eval_batch_size", "64", "--temperature", "28.0",
                "--lr", "0.005", "--lr_scale", "0.2",
                "--lr_scale_patterns", "patch_embed",
                "--null_eta1", "0.999", "--null_eta2", "0.999", "-tc", "2.0",
                "--eval-tool", "adapter",
                "--eval-trained-task-router", "true",
                "--eval-local-task-head", "true",
                "--eval-task-weight", "false",
                "--freeze-shared-prompts-after-first", "true",
                "--prototype-confidence-weight", "0.0",
                "--seed", str(args.seed),
                "--save-model", "false", "--logs-dir", "logs", "-sf", ckaa_suffix,
            ],
            ckaa_log, ckaa_suffix, args.stall, args.cap,
        )
        print(f"[final] ckaa: {outcome}", flush=True)

    # 3) Report.
    print("\n| run | Last-acc | Avg-acc | Forgetting | log |", flush=True)
    print("| --- | ---: | ---: | ---: | --- |", flush=True)
    for case in ("fp32", "fixed8", "fixed4", "fixed2"):
        path = ROOT / "logs" / f"{prefix}_{case}_s{args.seed}.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        results = RESULT_RE.findall(text)
        if len(results) < 5:
            print(f"| {case} | INVALID ({len(results)}/5 tasks) | - | - | {path.name} |")
            continue
        rows.append((case, float(results[-1][1]), float(results[-1][2]), float(results[-1][3]), path.name))
    if ckaa_log.exists():
        text = ckaa_log.read_text(encoding="utf-8", errors="replace")
        results = RESULT_RE.findall(text)
        if len(results) >= 5:
            rows.append(("paper1_ckaa", float(results[-1][1]), float(results[-1][2]), float(results[-1][3]), ckaa_log.name))
    for name, last, avg, forg, log in rows:
        print(f"| {name} | {last:.2f} | {avg:.2f} | {forg:.2f} | {log} |", flush=True)
    if rows:
        anchor = next((r for r in rows if r[0] == "fp32"), None)
        print("\nretention vs 32-bit:", flush=True)
        for name, last, _avg, _forg, _log in rows:
            if anchor and name != "fp32":
                print(f"  {name:12s} {last:.2f}  = {100 * last / anchor[1]:5.1f}%", flush=True)
    print("[final] done", flush=True)


if __name__ == "__main__":
    main()
