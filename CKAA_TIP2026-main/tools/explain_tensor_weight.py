"""Explain tensors and weights concretely, using this project's real shapes.

Nothing here is abstract: every shape below is an actual tensor of the tabular ViT
used in this project.
"""

from __future__ import annotations

import torch
from torch import nn


def main() -> None:
    print("=" * 92)
    print("(1) A TENSOR is just a multi-dimensional array of numbers")
    print("=" * 92)
    examples = [
        ("0-D  scalar      ", torch.tensor(3.14), "loss, one learning rate"),
        ("1-D  vector      ", torch.randn(768), "one bias, one feature vector"),
        ("2-D  matrix      ", torch.randn(768, 768), "one Linear layer's weight"),
        ("3-D  batch tensor", torch.randn(32, 196, 768), "activations of one layer"),
        ("4-D  image batch ", torch.randn(32, 4, 1568), "the raw fault signal batch"),
    ]
    for label, tensor, meaning in examples:
        print(f"  {label} shape={str(tuple(tensor.shape)):>18s}   e.g. {meaning}")

    print()
    print("  Our model's input: a batch of 32 windows, each 4 channels x 1568 samples")
    print("    -> tensor shape [32, 4, 1568]")
    print("  After patch embedding (Conv1d, kernel 32, stride 32):")
    print("    -> tensor shape [32, 49, 768]   (49 patches of 768 features)")
    print("  The CKAA tabular tokenizer pads to 196 patches:")
    print("    -> tensor shape [32, 196, 768]  (or 197 with the CLS token)")

    print()
    print("=" * 92)
    print("(2) A WEIGHT is a tensor that is a LEARNED PARAMETER")
    print("=" * 92)
    linear = nn.Linear(768, 768)
    print(f"  nn.Linear(768, 768) holds two parameter tensors:")
    for name, parameter in linear.named_parameters():
        print(
            f"    {name:8s} shape={str(tuple(parameter.shape)):>12s} "
            f"numel={parameter.numel():>8,d}  requires_grad={parameter.requires_grad}"
        )
    print()
    print("  'weight' = the matrix that multiplies the input (the thing training learns)")
    print("  'bias'   = an additive offset, also learned")
    print()
    print("  A layer computes:  y = x @ W.T + b")
    print("     x : input tensor  (computed, changes every batch  -> ACTIVATION)")
    print("     W : weight tensor (learned, constant per batch     -> WEIGHT)")
    print("     b : bias tensor   (learned)")

    print()
    print("=" * 92)
    print("(3) Weight vs activation: the key distinction for quantization")
    print("=" * 92)
    print(f"  {'':16s} {'weight':38s} {'activation':38s}")
    print(f"  {'origin':16s} {'learned by gradient descent':38s} {'computed by forward()':38s}")
    print(f"  {'lifetime':16s} {'persists across batches':38s} {'exists only during a forward':38s}")
    print(f"  {'range':16s} {'exact, cheap to compute':38s} {'estimated, needs calibration':38s}")
    print(f"  {'stored?':16s} {'yes, in the .pth':38s} {'no, recomputed each time':38s}")
    print(f"  {'size driver?':16s} {'YES - determines model size':38s} {'no - only affects compute':38s}")

    print()
    print("=" * 92)
    print("(4) Where the ~90.9 M numbers of this model live")
    print("=" * 92)
    rows = [
        ("ViT blocks (12 x attn.qkv)", 2304 * 768 * 12),
        ("ViT blocks (12 x attn.proj)", 768 * 768 * 12),
        ("ViT blocks (12 x mlp.fc1)", 3072 * 768 * 12),
        ("ViT blocks (12 x mlp.fc2)", 768 * 3072 * 12),
        ("patch_embed Conv1d", 4 * 768 * 32),
        ("prompts / adapters / heads", 5_900_000),
    ]
    total = 0
    for name, count in rows:
        total += count
        print(f"  {name:32s} {count:>13,d}")
    print(f"  {'-' * 46}")
    print(f"  {'TOTAL weight scalars':32s} {total:>13,d}")
    print()
    print("  Every one of these is quantized: at 8-bit each needs 1 byte,")
    print("  at 4-bit half a byte, at 2-bit a quarter of a byte.")


if __name__ == "__main__":
    main()
