"""Measure the cost of the range observation, with and without quantile clipping.

Motivation: `_observed_range` ran `torch.quantile` (a sort per channel) on every
forward call, while the quantile is only needed when the range is actually being
re-calibrated. This benchmark sizes that cost at the real activation shape.
"""

from __future__ import annotations

import os.path as osp
import statistics
import sys
import time

import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.aqcl import UniformQuantizer  # noqa: E402


def bench(clip: float, reduce: str, x: torch.Tensor, reps: int = 5, iters: int = 10) -> float:
    quantizer = UniformQuantizer(2, per_channel=(reduce == "channel"), range_mode="ema")
    quantizer.reduce = reduce
    quantizer.clip_percentile = clip
    quantizer._finitude_check = False
    for _ in range(2):
        quantizer._observed_range(x)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            quantizer._observed_range(x)
        times.append((time.perf_counter() - start) / iters * 1000)
    return statistics.median(times)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 197, 768) * 3.0
    print(f"shape {tuple(x.shape)}")
    print(f"  amin/amax, per-channel, clip=0     : {bench(0.0, 'channel', x):9.2f} ms")
    print(f"  quantile,  per-channel, clip=0.999 : {bench(0.999, 'channel', x):9.2f} ms")
    print(f"  quantile,  per-tensor,  clip=0.999 : {bench(0.999, 'all', x):9.2f} ms")


if __name__ == "__main__":
    main()
