"""Explain the `sd` column: what it is, and how large the seed spread really is.

`sd` is the sample standard deviation (n-1 denominator) of Last-acc across the
seeds run for one configuration. It answers "if I re-run this configuration with a
different seed, how far from the mean should I expect the number to land?" - which
is exactly the uncertainty that was ignored when single-seed rows were compared.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LAST_RE = re.compile(r"Results of task \[5\]: \[ class_inc_last_acc=([\d.]+)%")

RUNS = {
    "32-bit": "fp32",
    "8-bit": "fixed8",
    "4-bit": "fixed4",
    "2-bit": "fixed2",
}
SEEDS = [2033, 2034, 2035, 2036]


def main() -> None:
    print(f"{'width':8s} {'n':>2s} {'mean':>7s} {'sd':>6s} {'CV':>6s} "
          f"{'min':>7s} {'max':>7s} {'range':>6s}  values")
    print("-" * 104)
    for label, case in RUNS.items():
        values = []
        for seed in SEEDS:
            path = ROOT / "logs" / f"U_{case}_s{seed}.txt"
            if not path.exists():
                continue
            found = LAST_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
            if found:
                values.append(float(found[-1]))
        if not values:
            continue
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = 100.0 * sd / mean if mean else 0.0
        print(
            f"{label:8s} {len(values):2d} {mean:7.2f} {sd:6.2f} {cv:5.1f}% "
            f"{min(values):7.2f} {max(values):7.2f} {max(values) - min(values):6.2f}  "
            f"{[round(v, 2) for v in values]}"
        )
    print("-" * 104)
    print("sd  = sample standard deviation (n-1) of Last-acc over the seeds")
    print("CV  = sd / mean, the spread as a percentage of the level")
    print("range = max - min, i.e. how far two seeds can differ")
    print()
    print("Reading: the seeds move the result by 4-13 points depending on the width,")
    print("so a single-seed row cannot be compared against another single-seed row.")
    print("That was the source of the earlier false claim that 4-bit was 18 points")
    print("below the paper: it compared one seed of 4-bit against a 4-seed mean.")


if __name__ == "__main__":
    main()
