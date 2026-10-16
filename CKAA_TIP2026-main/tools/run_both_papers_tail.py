"""Wait for the AQCL 32-8-4-2 sweep, then run the CKAA (paper 1) baseline.

Both papers then have a fresh run from the current code in one serial chain. The
CKAA run must use train_eval.py unchanged; only the AQCL runs may use train_aqcl.py.

Usage: python tools/run_both_papers_tail.py --prefix r24 --seed 2024
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent / "_aqcl_work"


def log_has_all_tasks(path: Path, tasks: int = 5) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"Results of task \[", text)) >= tasks


def wait_for_run(path: Path, stall_seconds: int, max_seconds: int) -> str:
    """Wait until the log is complete or stops growing."""
    start = time.time()
    last_size, last_change = -1, time.time()
    while True:
        if log_has_all_tasks(path):
            return "complete"
        now = time.time()
        size = path.stat().st_size if path.exists() else 0
        if size != last_size:
            last_size, last_change = size, now
        if now - last_change > stall_seconds:
            return "stalled"
        if now - start > max_seconds:
            return "max"
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="r24")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()

    # 1) Wait for the AQCL sweep to finish.
    for case in ("fp32", "fixed8", "fixed4", "fixed2"):
        path = ROOT / "logs" / f"{args.prefix}_{case}_s{args.seed}.txt"
        outcome = wait_for_run(path, args.stall_seconds, args.max_seconds)
        print(f"[chain] aqcl {case}: {outcome}", flush=True)

    # 2) Run the CKAA baseline (paper 1) with the unmodified entry point.
    ckaa_log = ROOT / "logs" / f"FINAL2_paper1_ckaa_s{args.seed}.txt"
    if log_has_all_tasks(ckaa_log):
        print("[chain] ckaa already complete", flush=True)
    else:
        command = [
            sys.executable, "-u", str(ROOT / "train_eval.py"),
            "-d", "fault_csv", "-t", "5",
            "-m", "vit_base_patch16_224.augreg_in21k",
            "-b", "32", "-e", "12", "-jt", "0", "-je", "0", "-et", "1",
            "--eval_batch_size", "64", "--temperature", "28.0",
            "--lr", "0.005", "--lr_scale", "0.2", "--lr_scale_patterns", "patch_embed",
            "--null_eta1", "0.999", "--null_eta2", "0.999", "-tc", "2.0",
            "--eval-tool", "adapter",
            "--eval-trained-task-router", "true",
            "--eval-local-task-head", "true",
            "--eval-task-weight", "false",
            "--freeze-shared-prompts-after-first", "true",
            "--prototype-confidence-weight", "0.0",
            "--seed", str(args.seed),
            "--save-model", "false", "--logs-dir", "logs",
            "-sf", f"FINAL2_paper1_ckaa_s{args.seed}",
        ]
        print("[chain] running CKAA (paper 1)", flush=True)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        with open(WORK / "ckaa_final2.out", "wb") as out, open(
            WORK / "ckaa_final2.err", "wb"
        ) as err:
            process = subprocess.Popen(
                command, cwd=str(ROOT), env=env, stdout=out, stderr=err
            )
            outcome = wait_for_run(ckaa_log, args.stall_seconds, args.max_seconds)
            if outcome != "complete":
                process.kill()
                process.wait()
        print(f"[chain] ckaa: {outcome}", flush=True)

    print("[chain] done", flush=True)


if __name__ == "__main__":
    main()
