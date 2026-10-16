"""Show exactly WHAT continuous values are being quantized in this model.

Quantization replaces a continuous float value with the nearest level on a grid.
Two different kinds of tensor get quantized:

  * WEIGHTS     - the learned parameters (they are constant during a forward pass,
                  so their range is exact and computed once);
  * ACTIVATIONS - the intermediate tensors produced inside forward(), i.e. every
                  input and output that flows between layers.

This script prints the real tensors of the tabular ViT used in this project, with
their shapes and (for activations) the ranges recorded in the logs, so the abstract
"continuous value x" becomes concrete.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shapes of the quantized layers in the tabular ViT-Base/16.
WEIGHT_SHAPES = {
    "patch_embed.proj (Conv1d)": "4 x 768 x 32",
    "blocks.*.attn.qkv (Linear)": "2304 x 768",
    "blocks.*.attn.proj (Linear)": "768 x 768",
    "blocks.*.mlp.fc1 (Linear)": "3072 x 768",
    "blocks.*.mlp.fc2 (Linear)": "768 x 3072",
    "blocks.*.prompt_linear": "768 x 768",
    "blocks.*.task_adapter.N.down_proj": "64 x 768",
    "blocks.*.task_adapter.N.up_proj": "768 x 64",
    "task_head.N (Linear)": "N_classes x 768",
}

ACT_RANGE_RE = re.compile(
    r"\[AQCL-INSTRUMENT\]\[act_range\] task=(\d+) epoch=(\d+) (\S+) "
    r"min=(-?[\d.]+) max=(-?[\d.]+) step=([\d.]+)"
)


def main() -> None:
    print("=" * 96)
    print("(A) WEIGHTS being quantized - constant tensors, exact range")
    print("=" * 96)
    print(f"  {'layer':38s} {'shape':>18s}   what the value is")
    for name, shape in WEIGHT_SHAPES.items():
        print(f"  {name:38s} {shape:>18s}   one learned scalar parameter")
    print()
    print("  Total quantized weight scalars: 187 layers, ~90.9 M parameters.")
    print("  Example: blocks.0.attn.qkv.weight is a 2304 x 768 float32 matrix = "
          "1,769,472 continuous values,")
    print("  each of which is independently snapped to a level of the N-bit grid.")

    print()
    print("=" * 96)
    print("(B) ACTIVATIONS being quantized - tensors produced inside forward()")
    print("=" * 96)
    print("  The activation quantizer sits on the INPUT of every quantized layer,")
    print("  so what it quantizes is the tensor flowing between two layers.")
    print()
    print(f"  {'layer':34s} {'tensor shape':>18s} {'min':>10s} {'max':>10s} {'span':>10s}")
    log = ROOT / "logs" / "U_fixed4_s2033.txt"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        seen = set()
        for match in ACT_RANGE_RE.finditer(text):
            name = match.group(3)
            key = name.split(".")[0] + name.split(".")[-1]
            if key in seen:
                continue
            seen.add(key)
            lo, hi = float(match.group(4)), float(match.group(5))
            print(f"  {name:34s} {'[B, 196, C]':>18s} {lo:10.4f} {hi:10.4f} {hi - lo:10.4f}")
            if len(seen) >= 8:
                break

    print()
    print("  For the tabular ViT the activation shapes are:")
    print("    patch_embed.proj            : [32, 196, 768]  (batch, patches, channels)")
    print("    blocks.*.attn.qkv / mlp.fc1 : [32, 197, 768]  (incl. the CLS token)")
    print("    blocks.*.mlp.fc1 pre-GELU   : [32, 197, 3072] (the widest tensor)")
    print("    task_head.N                 : [32, 197, 768]")
    print()
    print("=" * 96)
    print("So 'the continuous value x' in the quantization formula is:")
    print("  * for weights     : one scalar of one weight matrix (a learned parameter)")
    print("  * for activations : one scalar of one intermediate tensor (a computed value)")
    print("=" * 96)


if __name__ == "__main__":
    main()
