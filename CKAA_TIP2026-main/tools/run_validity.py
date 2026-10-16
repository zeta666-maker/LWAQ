"""Shared validity guard for AQCL training runs.

A training run only counts as a measurement of its configuration when *all* of
the following hold:

* every expected task printed a ``Results of task [i]`` line;
* no epoch reported ``loss=nan`` (numerical divergence);
* a deployment line with a finite model size was printed.

Without this guard an incomplete or diverged log is parsed as a finished run and
its numbers enter a comparison table. That happened in this project: the log
``logs/p2_fixed2_s2024.txt`` diverged on task 3 and finished only 2 tasks, yet it
was reported as a 2-bit accuracy of 58.86% (which was really the task-2 accuracy
of a collapsed model), while the healthy 2-bit run of the same seed
(``logs/tc_fixed2_s2024.txt``) reached only 42.55%. Any conclusion drawn from the
first number was invalid.

Usage:
    from run_validity import check_run
    report = check_run("logs/p2_fixed2_s2024.txt", expected_tasks=5)
    if not report.ok:
        print(report.reason)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path

TASK_RE = re.compile(r"Results of task \[(\d+)\]")
NAN_RE = re.compile(r"loss=nan")
SIZE_RE = re.compile(r"AQCL deployment: size=([\d.]+)MB")


@dataclass
class RunValidity:
    path: str
    ok: bool
    reason: str
    tasks_done: int
    nan_epochs: int
    size_mb: float | None
    notes: list[str] = field(default_factory=list)


def check_run(
    path: str | Path, expected_tasks: int = 5, quantized: bool = True
) -> RunValidity:
    """Return whether a log represents a complete, finite run.

    ``quantized=False`` (the fp32 baseline) has no quantizer and therefore prints
    no deployment line; its size must come from the full-precision parameter
    count instead.
    """
    path = Path(path)
    if not path.exists():
        return RunValidity(str(path), False, "log missing", 0, 0, None)
    text = path.read_text(encoding="utf-8", errors="replace")
    tasks = len(TASK_RE.findall(text))
    nan_epochs = len(NAN_RE.findall(text))
    sizes = SIZE_RE.findall(text)

    problems: list[str] = []
    if nan_epochs:
        problems.append(f"diverged ({nan_epochs} epoch(s) with loss=nan)")
    if tasks < expected_tasks:
        problems.append(f"incomplete ({tasks}/{expected_tasks} tasks)")
    if not sizes and quantized:
        problems.append("no deployment line")
    # Cross-check the deployment-line count against the task count: one
    # deployment line is printed per task. A result whose own log does not
    # contain the run it claims (seen when two runs write the same filename)
    # must not be trusted.
    if quantized and sizes and len(sizes) < tasks:
        problems.append(
            f"internal inconsistency ({len(sizes)} deployment lines for {tasks} tasks)"
        )

    return RunValidity(
        path=str(path),
        ok=not problems,
        reason="; ".join(problems) if problems else "ok",
        tasks_done=tasks,
        nan_epochs=nan_epochs,
        size_mb=float(sizes[-1]) if sizes else None,
    )


def load_metrics(path: str | Path, expected_tasks: int = 5) -> dict | None:
    """Parse a log's final metrics, or ``None`` when the run is not valid.

    Callers must treat ``None`` as "no measurement", never as a low score.
    """
    validity = check_run(path, expected_tasks=expected_tasks)
    if not validity.ok:
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    results = re.findall(
        r"Results of task \[(\d+)\]: \[ class_inc_last_acc=([\d.]+)% \| "
        r"class_inc_avg_acc=([\d.]+)% \| class_inc_last_forg=\s*([\d.]+)% \]",
        text,
    )
    last = results[-1]
    return {
        "task": int(last[0]),
        "last_acc": float(last[1]),
        "avg_acc": float(last[2]),
        "forgetting": float(last[3]),
        "size_mb": validity.size_mb,
        "validity": validity,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--tasks", type=int, default=5)
    args = parser.parse_args()
    for path in args.logs:
        validity = check_run(path, expected_tasks=args.tasks)
        status = "VALID  " if validity.ok else "INVALID"
        print(
            f"{status} {Path(path).name:44s} tasks={validity.tasks_done} "
            f"nan={validity.nan_epochs} size={validity.size_mb} :: {validity.reason}"
        )


if __name__ == "__main__":
    main()
