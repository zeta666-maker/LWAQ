"""List exactly which parameters CKAA trains, and how many there are.

Uses the same `set_model_mode` logic as train_eval.py so the answer comes from the
real code path, not from documentation.
"""

from __future__ import annotations

import os.path as osp
import sys
from collections import defaultdict

import torch
from torch import nn

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_eval as ckaa  # noqa: E402


def component_of(name: str) -> str:
    """Group a parameter name into a human-readable component (ASCII labels)."""
    if "patch_embed" in name:
        return "patch_embed (Conv1d tokenizer)"
    if "sh_prompt" in name:
        return "sh_prompt (shared prompt)"
    if "sp_prompt" in name:
        return "sp_prompt (per-task prompt)"
    if "task_adapter" in name:
        return "task_adapter (per-task adapter)"
    if "task_head" in name:
        return "task_head (per-task classifier)"
    if "prompt_linear" in name or "prompt" in name:
        return "prompt_linear"
    if "keys" in name:
        return "keys (router/prototypes)"
    if "head" in name:
        return "head (main classifier)"
    return "ViT backbone (frozen)"


def main() -> None:
    sys.argv = [
        sys.argv[0],
        "-d", "fault_csv", "-t", "5",
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32", "-jt", "0", "-je", "0", "-et", "1",
        "--logs-dir", "logs", "-sf", "trainable_probe",
        "--freeze-shared-prompts-after-first", "true",
    ]
    args = ckaa.get_args()
    ckaa.args = args
    ckaa.seed_etc_options(args.seed)

    import timm
    from utils import misc

    GVM = ckaa.GlobalVarsManager()
    GVM.init_from_args(args)
    taskid, current = next(iter(GVM.cl_mngr))
    GVM.update_label_maps(taskid, current)
    ckaa.current_task_classes = current
    GVM.cache_dict["training_string"] = args.training_string

    model = timm.create_model(
        args.model,
        pretrained=False,
        pretrained_strict=False,
        **ckaa.get_head_dim_arg_dict(GVM, args),
        prompt_args_dict=misc.get_specific_args_dict(args, "prompt_"),
        other_args_dict=misc.get_specific_args_dict(args, "logit_"),
    )
    model = ckaa.convert_vit_to_tabular(
        model,
        signal_length=args.tabular_window_length,
        patch_size=args.tabular_patch_size,
        in_chans=args.tabular_in_chans,
    )
    # train_eval wraps the model in DataParallel before touching the head.
    model = nn.DataParallel(model, device_ids=[0])
    model = ckaa.modify_head(GVM, model, training=True, task_classes=current)
    model = ckaa.set_model_mode(
        GVM, taskid, model, training=True, training_string=args.training_string
    )

    print(f"training_string = {args.training_string}")
    print()
    counts = defaultdict(lambda: [0, 0])  # component -> [trainable, frozen]
    examples = defaultdict(list)
    for name, parameter in model.named_parameters():
        bucket = 0 if parameter.requires_grad else 1
        group = component_of(name)
        counts[group][bucket] += parameter.numel()
        if len(examples[group]) < 2:
            examples[group].append(f"{name} {tuple(parameter.shape)}")

    total_train = sum(v[0] for v in counts.values())
    total_all = total_train + sum(v[1] for v in counts.values())

    print(f"{'component':40s} {'trainable':>14s} {'frozen':>14s}")
    print("-" * 70)
    for group, (train, frozen) in sorted(
        counts.items(), key=lambda kv: -kv[1][0]
    ):
        print(f"{group:40s} {train:>14,d} {frozen:>14,d}")
    print("-" * 70)
    print(f"{'TOTAL':40s} {total_train:>14,d} {total_all - total_train:>14,d}")
    print()
    print(f"trainable share = {100 * total_train / total_all:.2f}%  "
          f"({total_train / 1e6:.2f}M of {total_all / 1e6:.2f}M)")
    print()
    print("example parameter names per component:")
    for group, names in examples.items():
        for name in names:
            print(f"  [{group}] {name}")


if __name__ == "__main__":
    main()
