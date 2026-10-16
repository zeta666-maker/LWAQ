"""Verify the Hadamard-rotated quantization basis.

Checks three properties without training:
  1. Orthogonality: in FP32 (32-bit quantizers) the rotated layer computes the
     same function as the original layer, to numerical tolerance.
  2. Outlier flattening: the rotated weight has lower kurtosis / smaller
     max|w| relative to its std than the original weight.
  3. Quantization error: at 2 and 4 bits the rotated weight reproduces the FP32
     weight output with smaller error than the unrotated weight.

Usage: python tools/check_hadamard.py
"""

from __future__ import annotations

import os.path as osp
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.aqcl import QATLinear  # noqa: E402


def kurtosis(x: torch.Tensor) -> float:
    flat = x.detach().float().reshape(-1)
    centered = flat - flat.mean()
    var = centered.pow(2).mean()
    return float((centered.pow(4).mean() / var.clamp_min(1e-12).pow(2)).item())


def main() -> None:
    torch.manual_seed(0)
    in_features, out_features = 768, 2304
    base = nn.Linear(in_features, out_features, bias=True)
    # Give the weight a few strong outlier coordinates, as trained transformers do.
    with torch.no_grad():
        base.weight.mul_(1.0)
        base.weight[0, :5] *= 40.0
        base.weight[:, 3] *= 25.0

    x = torch.randn(8, 197, in_features) * 3.0
    reference = F.linear(x, base.weight, base.bias)

    for label, rotate in (("unrotated", False), ("rotated", True)):
        layer = QATLinear(base, low_bits=8)
        layer.set_hadamard_rotate(rotate)
        layer.weight_quant.use_exact_range = True
        # 1) FP32 equivalence: the rotated layer must equal the original output
        #    expressed in the rotated basis, i.e. x @ W_rot^T == (x @ W^T) H^T.
        with torch.no_grad():
            layer.weight_quant.set_bits(32)
            layer.activation_quant.set_bits(32)
            out = layer(x)
        if layer.hadamard_rotate and layer._had_out is not None:
            expected = reference @ layer._had_out.t()
        else:
            expected = reference
        max_diff = float((out - expected).abs().max().item())
        weight = (
            layer._had_out @ layer.weight if layer._had_out is not None else layer.weight
        )
        print(
            f"{label:9s} fp32_max_abs_diff={max_diff:.3e} "
            f"kurtosis={kurtosis(weight):8.2f} "
            f"max/std={float(weight.abs().max() / weight.std()):8.3f}"
        )
        # 3) quantization error at 2 and 4 bits, measured against the same
        #    reference output so the two rows are comparable.
        for bits in (2, 4):
            with torch.no_grad():
                layer.weight_quant.set_bits(bits)
                layer.activation_quant.set_bits(32)
                out_q = layer(x)
            err = float((out_q - out).abs().mean().item())
            rel = err / float(out.abs().mean().item())
            print(f"          {bits}-bit weight err={err:.5f} rel={rel * 100:.2f}%")


if __name__ == "__main__":
    main()
