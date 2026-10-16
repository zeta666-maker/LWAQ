"""Profile one training step at a given bit-width to locate the slowdown.

Motivation: after the low-bit stability fixes, 2-bit training ran at ~110 s per
batch instead of ~16 s, reproducibly, with healthy GPU clocks (2610/3105 MHz,
70-75 W, 80 C, no throttle). The quantizer itself only measured ~5 ms per call, so
the cost must be elsewhere in the step. This script times the pieces directly.

Usage:
  python tools/profile_lowbit_step.py --bits 2 --steps 5
  python tools/profile_lowbit_step.py --bits 4 --steps 5
  python tools/profile_lowbit_step.py --bits 32 --steps 5
"""

from __future__ import annotations

import argparse
import os.path as osp
import statistics
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


def parse_extra():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--quantize", type=int, default=1)
    known, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    args = ckaa.get_args()
    ckaa.args = args
    return args, known


def main() -> None:
    args, probe = parse_extra()
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
        pretrained=True,
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
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[device.index or 0])
    model = ckaa.set_model_mode(
        GVM, taskid, model, training=True, training_string=args.training_string
    )
    model = ckaa.modify_head(GVM, model, training=True, task_classes=current)

    args.aqcl_mode = "fixed"
    args.aqcl_bits = probe.bits
    args.aqcl_low_bits = probe.bits
    args.aqcl_high_bits = probe.bits
    args.aqcl_backbone_bits = -1
    args.aqcl_allocation = "mean"
    args.aqcl_fisher_temperature = 64.0
    args.aqcl_per_token_activation = True
    args.aqcl_instrument = False

    context = AQCLContext(model.module, args)
    # train_aqcl moves the rebuilt head to the device before use; do the same here.
    model.module.head = model.module.head.to(device)
    context.sync_classifier_head(model.module)
    for module in context.targets.values():
        module.to(device)
    context.set_bits(
        {name: probe.bits for name in context.targets},
        {name: probe.bits for name in context.targets},
    )
    context.set_quantization_enabled(bool(probe.quantize))

    dataset = ckaa.define_dataset(
        GVM, current, training=True, transform_type=args.transform_type,
        target_map_to_local=args.seperate_head, expand_times=1,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    iterator = iter(loader)

    def one_step() -> tuple[float, float, float]:
        images, target, _ = next(iterator)
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, logits, _ = model(
            images, mode=args.train_tool + "_train", taskid=taskid, max_taskid=taskid
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        criterion(logits, target).backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    for _ in range(2):  # warmup
        one_step()
    fwds, bwds, steps = [], [], []
    for _ in range(probe.steps):
        f, b, s = one_step()
        fwds.append(f)
        bwds.append(b)
        steps.append(s)
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    print(
        f"bits={probe.bits:2d} quantize={bool(probe.quantize)!s:5s} "
        f"forward={statistics.median(fwds):8.1f} ms  "
        f"backward={statistics.median(bwds):8.1f} ms  "
        f"opt={statistics.median(steps):7.1f} ms  "
        f"total={statistics.median(fwds) + statistics.median(bwds) + statistics.median(steps):8.1f} ms  "
        f"params_finite={finite}"
    )


if __name__ == "__main__":
    main()
