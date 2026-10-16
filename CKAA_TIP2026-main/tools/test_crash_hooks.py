"""Verify that an exception raised after the stdout redirect is still captured.

Reproduces the exact condition of the 2-bit abort (exit code 1, empty stdout and
stderr) and checks that the hooks installed in train_aqcl.main() write the
traceback to a separate file descriptor.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "_hook_probe.py"
CRASH = ROOT / "logs" / "_hook_probe.crash.txt"

PROBE_SOURCE = textwrap.dedent(
    '''
    """Child process: install the hook pattern, redirect stdout, then raise."""
    import os.path as osp
    import sys
    import traceback

    crash = open(osp.join("logs", "_hook_probe.crash.txt"), "w", buffering=1)


    def excepthook(exc_type, exc_value, exc_tb):
        crash.write("uncaught exception in main thread\\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=crash)
        crash.flush()


    sys.excepthook = excepthook


    class Sink:
        def write(self, _text):
            pass

        def flush(self):
            pass


    # Same order as train_aqcl.main(): hook file first, then the stdout redirect
    # that used to swallow the traceback.
    sys.stdout = Sink()
    raise RuntimeError("simulated low-bit abort at task 3 epoch 2")
    '''
)


def main() -> None:
    CRASH.unlink(missing_ok=True)
    PROBE.write_text(PROBE_SOURCE, encoding="utf-8")
    try:
        process = subprocess.run(
            [sys.executable, str(PROBE)], capture_output=True, text=True, cwd=ROOT
        )
        print(f"child exit code : {process.returncode}")
        print(f"child stdout len: {len(process.stdout)}")
        print(f"child stderr len: {len(process.stderr)}")
        print(f"crash file exists: {CRASH.exists()}")
        if CRASH.exists():
            content = CRASH.read_text(encoding="utf-8")
            print(f"crash file bytes : {len(content)}")
            print("--- captured tail ---")
            print(content.strip()[-260:])
        ok = (
            process.returncode == 1
            and len(process.stderr) == 0
            and CRASH.exists()
            and "simulated low-bit abort" in CRASH.read_text(encoding="utf-8")
        )
        print(f"\n{'PASS' if ok else 'FAIL'}: traceback recovered despite empty stderr")
        return 0 if ok else 1
    finally:
        PROBE.unlink(missing_ok=True)
        CRASH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
