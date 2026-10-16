"""Run the AQCL comparison matrix on fault_csv.

This is the Python equivalent of ``tools/run_aqcl_comparison.ps1`` and exists
because this machine disables PowerShell script execution. It launches
``train_aqcl.py`` sequentially, one case after another, and skips cases whose
log already contains the final task result.

The CKAA entry point (``train_eval.py``) is never used or modified here.

Examples
--------
python tools/run_aqcl_matrix.py                       # full matrix, backbone floor 8
python tools/run_aqcl_matrix.py --cases fp32,fixed8   # subset
python tools/run_aqcl_matrix.py --backbone-bits 0 --prefix p4_paperfaithful
python tools/run_aqcl_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# case -> (aqcl-enable, mode, aqcl-bits, low-bits, high-bits)
CASES: dict[str, tuple[str, str, int, int, int]] = {
    "fp32": ("false", "fixed", 32, 32, 32),
    "fixed8": ("true", "fixed", 8, 8, 8),
    "fixed4": ("true", "fixed", 4, 4, 4),
    "fixed2": ("true", "fixed", 2, 2, 2),
    "rpq_8_16": ("true", "rpq", 8, 8, 16),
    "rpq_4_8": ("true", "rpq", 4, 4, 8),
    "rpq_saou_8_16": ("true", "rpq_saou", 8, 8, 16),
    "rpq_saou_4_8": ("true", "rpq_saou", 4, 4, 8),
    "rpq_saou_2_4": ("true", "rpq_saou", 2, 2, 4),
}

DEFAULT_CASES = (
    "fp32",
    "fixed8",
    "fixed4",
    "fixed2",
    "rpq_4_8",
    "rpq_saou_4_8",
    "rpq_saou_2_4",
)


def build_command(args, case: str) -> list[str]:
    enable, mode, bits, low, high = CASES[case]
    suffix = f"{args.prefix}_{case}"
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_aqcl.py"),
        "-d", "fault_csv",
        "-t", str(args.tasks),
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32",
        "-e", str(args.epochs),
        "-jt", "0",
        "-je", "0",
        "-et", "1",
        "--eval_batch_size", "64",
        "--temperature", "28.0",
        "--lr", str(args.lr),
        "--lr_scale", "0.2",
        "--lr_scale_patterns", "patch_embed",
        "--null_eta1", "0.999",
        "--null_eta2", "0.999",
        "-tc", "2.0",
        "--eval-tool", "adapter",
        "--eval-trained-task-router", "true",
        "--eval-local-task-head", "true",
        "--eval-task-weight", "false",
        "--prototype-confidence-weight", "0.0",
        "--seed", "2024",
        "--save-model", "false",
        "--logs-dir", "logs",
        "-sf", suffix,
        "--aqcl-enable", enable,
        "--aqcl-mode", mode,
        "--aqcl-bits", str(bits),
        "--aqcl-low-bits", str(low),
        "--aqcl-high-bits", str(high),
        "--aqcl-backbone-bits", str(args.backbone_bits),
        "--aqcl-fisher-batches", str(args.fisher_batches),
        "--aqcl-warmup-epochs", str(args.warmup_epochs),
        "--aqcl-calibrate-batches", str(args.calibrate_batches),
        "--aqcl-per-token-activation", str(args.per_channel_activation).lower(),
    ]
    if args.instrument:
        command += ["--aqcl-instrument", "true", "--aqcl-instrument-verbose", "false"]
    return command


def already_complete(log_path: Path, tasks: int) -> bool:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return f"Results of task [{tasks}]" in text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--prefix", default="aqcl_compare")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument(
        "--backbone-bits",
        type=int,
        default=8,
        help="Precision floor for the frozen ViT backbone (0 = paper-faithful).",
    )
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--per-channel-activation",
        action="store_true",
        default=True,
        help="Use one activation quantization range per feature channel "
        "(default on; it is a large accuracy win at 4-bit and 2-bit).",
    )
    parser.add_argument(
        "--per-tensor-activation",
        dest="per_channel_activation",
        action="store_false",
        help="Revert to one activation range per tensor (ablation).",
    )
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--fisher-batches", type=int, default=8)
    parser.add_argument("--calibrate-batches", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-case wall-clock timeout in seconds (0 = no timeout). A case "
        "that exceeds it is killed and the matrix continues.",
    )
    args = parser.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        raise SystemExit(f"Unknown cases {unknown}; known {sorted(CASES)}")

    failures: list[str] = []
    for case in cases:
        suffix = f"{args.prefix}_{case}"
        log_path = ROOT / "logs" / f"{suffix}.txt"
        if not args.force and already_complete(log_path, args.tasks):
            print(f"[matrix] skip {suffix} (already complete): {log_path}")
            continue
        command = build_command(args, case)
        print(f"[matrix] run {suffix}", flush=True)
        print("         " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Redirect the child's stdout/stderr to files. Letting it inherit the
        # parent's pipe can deadlock when the pipe buffer fills up (the child
        # blocks on write while the parent is not draining it).
        output_dir = ROOT.parent / "_aqcl_work"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{suffix}.out"
        err_path = output_dir / f"{suffix}.err"
        try:
            with open(out_path, "wb") as out_handle, open(
                err_path, "wb"
            ) as err_handle:
                result = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    env=env,
                    stdout=out_handle,
                    stderr=err_handle,
                    timeout=args.timeout if args.timeout > 0 else None,
                )
            code = result.returncode
        except subprocess.TimeoutExpired:
            print(f"[matrix] {suffix} TIMEOUT after {args.timeout}s", flush=True)
            failures.append(suffix)
            continue
        print(f"[matrix] {suffix} exit={code} stdout={out_path}", flush=True)
        if code != 0:
            failures.append(suffix)
    if failures:
        print(f"[matrix] failures: {failures}")
    print("[matrix] done")


if __name__ == "__main__":
    main()
