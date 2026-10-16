"""Benchmark the learnable-scale overhead in the quantized forward/backward.

Diagnostic only. Runs a few training steps with and without the LSQ-style
learnable step multiplier and reports the wall-clock time, so a slowdown can be
attributed instead of guessed.

Usage:
  python tools/bench_learnable_scale.py --steps 20
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import train_eval as ckaa  # noqa: E402
from utils.aqcl import AQCLContext  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--steps", type=int, default=20)
    known, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    args = ckaa.get_args()
    ckaa.args = args
    return args, known


def main():
    args, probe = parse_args()
    ckaa.seed_etc_options(args.seed)
    import timm
    from utils import misc

    GVM = ckaa.GlobalVarsManager()
    GVM.init_from_args(args)
    taskid, current = next(iter(GVM.cl_mngr))
    GVM.update_label_maps(taskid, current)
    ckaa.current_task_classes = current
    GVM.cache_dict["training_string"] = args.training_string

    prompt_args = misc.get_specific_args_dict(args, "prompt_")
    other_args = misc.get_specific_args_dict(args, "logit_")
    head_args = ckaa.get_head_dim_arg_dict(GVM, args)
    model = timm.create_model(
        args.model,
        pretrained=True,
        pretrained_strict=False,
        **head_args,
        prompt_args_dict=prompt_args,
        other_args_dict=other_args,
    )
    model = ckaa.convert_vit_to_tabular(
        model,
        signal_length=args.tabular_window_length,
        patch_size=args.tabular_patch_size,
        in_chans=args.tabular_in_chans,
    )
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[device.index or 0])
    model = ckaa.set_model_mode(
        GVM, taskid, model, training=True, training_string=args.training_string
    )
    model = ckaa.modify_head(GVM, model, training=True, task_classes=current)

    args.aqcl_low_bits = 4
    args.aqcl_high_bits = 4
    args.aqcl_bits = 4
    args.aqcl_mode = "fixed"
    args.aqcl_backbone_bits = -1
    args.aqcl_allocation = "mean"
    args.aqcl_fisher_temperature = 64.0
    args.aqcl_per_token_activation = True

    dataset = ckaa.define_dataset(
        GVM, current, training=True, transform_type=args.transform_type,
        target_map_to_local=args.seperate_head, expand_times=1,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss().to(device)

    for mode in (False, True, "graph_only"):
        args.aqcl_learnable_scale = mode is not False
        # Rebuild the quantizer wrappers each round so every buffer starts on the
        # model's device.
        model = ckaa.set_model_mode(
            GVM, taskid, model, training=True, training_string=args.training_string
        )
        context = AQCLContext(model.module, args)
        context.set_bits(
            {name: 4 for name in context.targets},
            {name: 4 for name in context.targets},
        )
        for module in context.targets.values():
            module.weight_quant.to(device)
            module.activation_quant.to(device)
            module.weight_quant.scale_raw = module.weight_quant.scale_raw.to(device)
            module.activation_quant.scale_raw = (
                module.activation_quant.scale_raw.to(device)
            )
        params = list(model.parameters()) + context.quantizer_parameters()
        if mode == "graph_only":
            # Keep the extra graph but do not hand the scalars to the optimizer,
            # to separate graph cost from many-small-parameters cost.
            params = list(model.parameters())
        optimizer = torch.optim.Adam(
            [{"params": [p for p in params if p.requires_grad], "lr": 1e-4}]
        )
        # warmup
        iterator = iter(loader)
        for _ in range(3):
            images, target, _ = next(iterator)
            images = images.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits, _ = model(
                images, mode=args.train_tool + "_train", taskid=taskid, max_taskid=taskid
            )
            criterion(logits, target).backward()
            optimizer.step()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(probe.steps):
            images, target, _ = next(iterator)
            images = images.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits, _ = model(
                images, mode=args.train_tool + "_train", taskid=taskid, max_taskid=taskid
            )
            criterion(logits, target).backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(
            f"learnable_scale={mode!s:5s} steps={probe.steps} "
            f"total={elapsed:.2f}s per_step={elapsed / probe.steps:.3f}s"
        )


if __name__ == "__main__":
    main()
