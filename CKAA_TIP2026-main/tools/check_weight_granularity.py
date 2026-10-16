"""Verify the granularity actually used for weight quantization (code vs comment).

The docstring in QATLinear says "weights never change between quantization steps"
and the code comments call the weight range "per-output-channel". The vector shapes
say otherwise, so this script measures which one it really is.

For a Linear layer, weight has shape [out_features, in_features].
  * per-OUTPUT-channel range -> shape [out_features, 1], reducing over in_features
  * per-INPUT-channel  range -> shape [in_features],      reducing over out_features
The two are NOT the same, and at low bit-width the choice matters.
"""

from __future__ import annotations

import os.path as osp
import sys

import torch

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.aqcl import QATLinear  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    out_features, in_features = 2304, 768
    layer = QATLinear(in_features, out_features, low_bits=4)
    weight = layer.weight.detach()
    quantizer = layer.weight_quant

    print(f"layer.weight shape          : {tuple(weight.shape)}")
    print(f"  out_features={out_features}, in_features={in_features}")
    print()

    low, high = quantizer._observed_range(weight)
    print(f"quantizer.reduce            : {quantizer.reduce!r}")
    print(f"quantizer.per_channel       : {quantizer.per_channel}")
    print(f"observed range shapes       : low={tuple(low.shape)} high={tuple(high.shape)}")
    print(f"numel of a range vector     : {low.numel()}")
    print()

    if low.numel() == in_features:
        verdict = "per-INPUT-channel  (reduces over out_features)"
    elif low.numel() == out_features:
        verdict = "per-OUTPUT-channel (reduces over in_features)"
    else:
        verdict = f"neither: {low.numel()} values"

    print(f"VERDICT: the weight range is {verdict}")
    print()

    # Demonstrate the consequence: a weight matrix whose rows differ in scale.
    # Per-input-channel reduction collapses rows of different scale into one range.
    weight2 = weight.clone()
    weight2[: out_features // 2] *= 50.0  # half the output channels are 50x larger
    low2, high2 = quantizer._observed_range(weight2)
    step2 = (high2 - low2) / 15.0
    per_out_span = (weight2.max(dim=1).values - weight2.min(dim=1).values)
    print("consequence demo: half the output channels scaled 50x")
    print(f"  per-input-channel step   : min {step2.min():.3e} max {step2.max():.3e}")
    print(f"  per-output-channel span  : min {per_out_span.min():.3e} "
          f"max {per_out_span.max():.3e}")
    print("  -> a per-output-channel range would give the small rows a much finer")
    print("     grid; a per-input-channel range forces the SAME step on both.")


if __name__ == "__main__":
    main()
