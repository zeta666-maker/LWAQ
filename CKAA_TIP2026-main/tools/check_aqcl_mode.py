"""Establish, from the logs, which AQCL components were ACTUALLY active.

Correction notice: an earlier version of this check looked only at the instrument
JSON's `num_layers_modulated` field and concluded "RPQ partially active, SAOU inert".
That was incomplete. The decisive fact is the `--aqcl-mode` actually passed on the
command line:

  fixed     -> uniform bit-width, NO RPQ, NO SAOU
  rpq       -> RPQ (sensitivity-aware bit allocation), no SAOU
  rpq_saou  -> RPQ + SAOU (the full paper method)

This script reads the command lines / arg echoes out of the logs so the answer comes
from what was run, not from what the default happens to be.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODE_RE = re.compile(r"--aqcl-mode[\s=]+(\w+)")
BITS_RE = re.compile(r"--aqcl-bits[\s=]+(\d+)")
BACKBONE_RE = re.compile(r"--aqcl-backbone-bits[\s=]+(-?\d+)")


def main() -> None:
    print("=" * 96)
    print("(1) What each --aqcl-mode actually enables")
    print("=" * 96)
    print("  fixed     : uniform bit-width everywhere. NO RPQ, NO SAOU.")
    print("  rpq       : + RPQ  (Eq.11-15 sensitivity-aware bit allocation)")
    print("  rpq_saou  : + RPQ + SAOU (Eq.16-24 gradient decomposition)  <- full paper")
    print()
    print("  parser default in train_aqcl.py: rpq_saou")
    print("  the runner tools/run_aqcl_tcmoa_matrix.py CASES table:")
    print('    "fixed2": ("true", "fixed", 2, 2, 2)   <- mode is "fixed"')

    print()
    print("=" * 96)
    print("(2) What the archived runs actually used")
    print("=" * 96)
    rows = []
    for path in sorted(glob.glob(str(ROOT / "logs" / "U_*.txt"))):
        name = os.path.basename(path)
        if "crash" in name or "faulthandler" in name:
            continue
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        mode = MODE_RE.search(text)
        bits = BITS_RE.search(text)
        backbone = BACKBONE_RE.search(text)
        rows.append(
            (
                name,
                mode.group(1) if mode else "?",
                bits.group(1) if bits else "?",
                backbone.group(1) if backbone else "?",
            )
        )

    if rows:
        print(f"  {'log':36s} {'mode':>10s} {'bits':>5s} {'backbone':>9s}")
        print("  " + "-" * 64)
        for name, mode, bits, backbone in rows:
            print(f"  {name:36s} {mode:>10s} {bits:>5s} {backbone:>9s}")
    else:
        print("  no --aqcl-mode string found in the U_* logs")

    print()
    modes = {r[1] for r in rows}
    print(f"  distinct modes across the 16 archived runs: {sorted(modes)}")

    print()
    print("=" * 96)
    print("(3) Consequence")
    print("=" * 96)
    if modes == {"fixed"}:
        print("  ALL archived runs used --aqcl-mode fixed.")
        print("  Therefore NEITHER of the paper's two contributions was active:")
        print("    * RPQ  (bit allocation)            : not active")
        print("    * SAOU (gradient decomposition)    : not active")
        print("  What the archived numbers measure is therefore:")
        print("    CKAA's continual-learning skeleton + UNIFORM N-bit quantization.")
        print("  They do NOT measure the AQCL method.")
    else:
        print(f"  modes found: {sorted(modes)} - inspect individually.")


if __name__ == "__main__":
    main()
