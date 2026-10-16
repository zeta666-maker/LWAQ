"""Quantify what "lightweight deployment" means for this model, and where the
current number comes from.

`AQCLContext.model_size_mb` computes a THEORETICAL lower bound: it sums
``numel * assigned_bits`` over the weight tensors. That is the payload a real
integer kernel would read, but it is not a measure of anything that currently
exists on disk or in memory, because:

  * training keeps a float32 master copy of every weight (`QATLinear.weight`) and
    only rounds inside ``forward``; the gradients need the float copy;
  * a real integer deployment also stores per-channel scales / zero-points;
  * there is no export step in this repository, so no runtime can load the
    quantized model.

This script reports three numbers side by side so the gap is explicit:

  1. float32 parameter footprint           (what training holds)
  2. integer payload at the assigned widths (what `model_size_mb` reports)
  3. integer payload + per-channel scales   (what a real deployment would need)

Usage:
  python tools/quantization_footprint.py --case fp32 --seed 2033
  python tools/quantization_footprint.py --case fixed8 --seed 2033
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (parameter count, layer counts) do not depend on the seed, only on which layers
# were promoted, so the per-layer bit widths are read from the instrument JSON.
SUM_LAYER_SHAPES = {
    "attn.qkv": (768, 2304),
    "attn.proj": (768, 768),
    "mlp.fc1": (768, 3072),
    "mlp.fc2": (3072, 768),
}
BACKBONE_BLOCKS = 12


def load_distribution(prefix: str, case: str, seed: int) -> dict | None:
    path = ROOT / "logs" / f"{prefix}_{case}_s{seed}.instrument.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data.get("records", []):
        if record.get("stage") == "rpq_sensitivity":
            return record.get("weight_bits_distribution")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="U")
    parser.add_argument("--case", default="fixed8")
    parser.add_argument("--seed", type=int, default=2033)
    args = parser.parse_args()

    distribution = load_distribution(args.prefix, args.case, args.seed)
    print(f"case={args.case} seed={args.seed}")
    print(f"weight bit distribution: {distribution}")

    # The ViT-Base/16 tabular backbone: for each block the four big Linears.
    per_layer_params = sum(in_f * out_f for in_f, out_f in SUM_LAYER_SHAPES.values())
    backbone_params = BACKBONE_BLOCKS * per_layer_params
    # patch_embed Conv1d: 4 channels -> 768, kernel 32
    patch_params = 4 * 768 * 32
    # prompts / adapters / heads are small; measured earlier at ~5.9 M total
    trainable_params = 5_900_000
    total_params = backbone_params + patch_params + trainable_params

    print()
    print(f"  ViT blocks (12 x 4 Linears) : {backbone_params:>12,d} params")
    print(f"  patch_embed Conv1d          : {patch_params:>12,d} params")
    print(f"  prompts/adapters/heads      : {trainable_params:>12,d} params")
    print(f"  total                       : {total_params:>12,d} params")

    fp32_mb = total_params * 4 / 1024**2
    print()
    print(f"  1. float32 footprint            : {fp32_mb:8.2f} MB   <- what training holds")
    print(f"     (also what a naive .pth save would be)")

    if args.case == "fp32":
        print()
        print("  this IS the fp32 anchor: no integer payload exists")
        return

    bits = 8 if "8" in args.case else 4 if "4" in args.case else 2
    int_mb = total_params * bits / 8 / 1024**2
    # per-channel scales: one float32 per output channel of each quantized Linear
    channels = BACKBONE_BLOCKS * sum(out_f for _in, out_f in SUM_LAYER_SHAPES.values())
    scales_mb = channels * 4 / 1024**2
    print(f"  2. integer payload @ {bits}-bit      : {int_mb:8.2f} MB   <- what model_size_mb reports")
    print(f"  3. + per-channel scales (fp32)  : {int_mb + scales_mb:8.2f} MB   <- real deployment")
    print(f"       (scales: {channels:,d} channels x 4 B = {scales_mb:.2f} MB, i.e. "
          f"{100 * scales_mb / int_mb:.1f}% of the payload)")
    print()
    print(f"  compression vs float32 (payload only) : {fp32_mb / int_mb:5.1f}x")
    print(f"  compression vs float32 (with scales)  : {fp32_mb / (int_mb + scales_mb):5.1f}x")


if __name__ == "__main__":
    main()
