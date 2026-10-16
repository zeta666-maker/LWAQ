"""Step-by-step walkthrough of how one number is quantized, with real values.

Covers both cases (a weight scalar and an activation scalar) and shows every
intermediate quantity, so the abstract formula becomes arithmetic.
"""

from __future__ import annotations

import os.path as osp
import sys

import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.aqcl import QATLinear  # noqa: E402


def walk(label: str, value: float, low: float, high: float, bits: int) -> float:
    levels = (1 << bits) - 1
    step = (high - low) / levels
    index = round((value - low) / step)
    quantized = index * step + low
    print(f"  {label}")
    print(f"    input value x            = {value:+.8f}")
    print(f"    range  [min, max]        = [{low:+.8f}, {high:+.8f}]")
    print(f"    span = max - min         = {high - low:.8e}")
    print(f"    bits N                   = {bits}  -> {1 << bits} levels ({levels} intervals)")
    print(f"    step n = span/(2^N - 1)  = {step:.8e}")
    print(f"    index = round((x-min)/n) = round({(value - low) / step:+.4f}) = {index}")
    print(f"    quantized = idx*n + min  = {quantized:+.8f}")
    print(f"    rounding error           = {quantized - value:+.8e}"
          f"   (max possible = n/2 = {step / 2:.3e})")
    return quantized


def main() -> None:
    torch.manual_seed(0)

    print("=" * 94)
    print("CASE 1: a WEIGHT scalar, 4-bit")
    print("=" * 94)
    layer = QATLinear(768, 2304, low_bits=4)
    weight = layer.weight.detach()
    quantizer = layer.weight_quant
    low, high = quantizer._observed_range(weight)
    # weight[out, in] -> range indexed by the input channel
    out_index, in_index = 0, 0
    walk(
        f"weight[{out_index},{in_index}] of a [2304, 768] matrix",
        float(weight[out_index, in_index]),
        float(low[in_index]),
        float(high[in_index]),
        4,
    )

    print()
    print("=" * 94)
    print("CASE 2: the same weight at 2-bit (what the failing runs used)")
    print("=" * 94)
    walk(
        "lowest-magnitude weight element",
        float(weight.abs().min()),
        float(low[in_index]),
        float(high[in_index]),
        2,
    )
    print("    note: with only 4 levels, any value is at most 1.5 steps from a level;")
    print("          the whole matrix collapses onto 4 distinct numbers per channel.")

    print()
    print("=" * 94)
    print("CASE 3: an ACTIVATION scalar, 4-bit (the tensor that actually diverged)")
    print("=" * 94)
    # The value and range recorded for the classifier head input in a real run.
    value = 3.0
    walked = walk(
        "one element of the head input tensor [32, 197, 768]",
        value,
        -16.7,
        16.7,
        4,
    )
    print()
    print("  why this case diverges at 2-bit, with the span measured in a real run")
    for bits in (8, 4, 2):
        step = 33.35 / ((1 << bits) - 1)
        print(f"    bits={bits:2d}: step = 33.35/{((1 << bits) - 1):>3d} = {step:9.5f}"
              f"   -> the 768-d input is represented by {1 << bits:>3d} distinct values")
    print()
    print("  The head input spans ~33 but 2-bit has only 4 levels, so ~4.8 M values")
    print("  are forced onto 4 numbers separated by ~11.1. That is the collapse.")


if __name__ == "__main__":
    main()
