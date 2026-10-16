"""Measure per-layer quantization error in the calibrated AQCL model.

Diagnostic only. Reports, for one calibrated model at a given bit-width:

  * activation outlier ratio: |x| beyond the calibrated range
  * activation signal-to-quantization-noise ratio (SQNR)
  * weight SQNR per layer
  * the resulting output logit shift versus the FP32 reference

This tells us whether low-bit accuracy loss comes from the quantization grid
itself or from something else in the training loop.

Usage:
  python tools/aqcl_quant_error.py -d fault_csv -t 5 -b 32 --bits 8 --warmup-epochs 0
"""

from __future__ import annotations

import argparse
import json
import math
import os.path as osp
import sys

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
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--out", type=str, default="")
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
    old = getattr(ckaa, "current_task_classes", None)
    ckaa.current_task_classes = current
    head_args = ckaa.get_head_dim_arg_dict(GVM, args)
    if old is not None:
        ckaa.current_task_classes = old
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[device.index or 0])
    model = ckaa.set_model_mode(
        GVM, taskid, model, training=True, training_string=args.training_string
    )
    model = ckaa.modify_head(GVM, model, training=True, task_classes=current)

    args.aqcl_low_bits = probe.bits
    args.aqcl_high_bits = probe.bits
    args.aqcl_bits = probe.bits
    args.aqcl_mode = "fixed"
    args.aqcl_backbone_bits = 0
    context = AQCLContext(model.module, args)
    context.set_bits(
        {name: probe.bits for name in context.targets},
        {name: probe.bits for name in context.targets},
    )

    dataset = ckaa.define_dataset(
        GVM,
        current,
        training=True,
        transform_type=args.transform_type,
        target_map_to_local=args.seperate_head,
        expand_times=1,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    stats = context.calibrate_ranges(
        model.module, dataloader, taskid, max_batches=probe.batches, freeze=True
    )

    # Collect per-layer SQNR over evaluation batches, quantized vs FP32.
    records: dict[str, dict] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs):
            x = inputs[0].detach().float()
            quantizer = module.activation_quant
            if int(quantizer.range_ready.item()) == 0:
                return
            lo = quantizer.range_min.float()
            hi = quantizer.range_max.float()
            step = ((hi - lo) / float((1 << quantizer.num_bits) - 1)).clamp_min(1e-12)
            clipped = ((x < lo) | (x > hi)).float().mean().item()
            noise_var = (step**2) / 12.0
            signal_var = x.var(unbiased=False).item()
            sqnr = 10.0 * math.log10(max(signal_var, 1e-12) / max(noise_var.item(), 1e-18))
            entry = records.setdefault(
                name, {"clipped": [], "sqnr": [], "step": [], "absmax": []}
            )
            entry["clipped"].append(clipped)
            entry["sqnr"].append(sqnr)
            entry["step"].append(float(step.item()))
            entry["absmax"].append(float(x.abs().max().item()))
        return hook

    for name, module in context.targets.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index >= probe.batches:
                break
            model(
                batch[0].to(device),
                mode=args.train_tool + "_train",
                taskid=taskid,
                max_taskid=taskid,
            )
    for handle in handles:
        handle.remove()

    rows = []
    for name, entry in records.items():
        module = context.targets[name]
        wq = module.weight.detach().float()
        w_lo = module.weight_quant.range_min.float()
        w_hi = module.weight_quant.range_max.float()
        w_step = ((w_hi - w_lo) / float((1 << probe.bits) - 1)).clamp_min(1e-12)
        w_noise = (w_step**2).mean() / 12.0
        w_var = wq.var(unbiased=False)
        w_sqnr = 10.0 * math.log10(
            max(float(w_var.item()), 1e-12) / max(float(w_noise.item()), 1e-18)
        )
        rows.append(
            {
                "layer": name,
                "activation_clipped_frac": sum(entry["clipped"]) / len(entry["clipped"]),
                "activation_sqnr_db": sum(entry["sqnr"]) / len(entry["sqnr"]),
                "activation_step": sum(entry["step"]) / len(entry["step"]),
                "activation_absmax": max(entry["absmax"]),
                "weight_sqnr_db": w_sqnr,
                "weight_step_mean": float(w_step.mean().item()),
                "weight_numel": int(module.weight.numel()),
            }
        )

    rows.sort(key=lambda r: r["activation_sqnr_db"])
    print(f"bits={probe.bits} batches={stats['num_batches']} layers={len(rows)}")
    print(
        f"{'layer':<44}{'clip%':>8}{'actSQNR':>10}{'wSQNR':>10}{'step':>10}{'absmax':>10}"
    )
    for row in rows[:16]:
        print(
            f"{row['layer']:<44}{row['activation_clipped_frac']*100:>8.2f}"
            f"{row['activation_sqnr_db']:>10.2f}{row['weight_sqnr_db']:>10.2f}"
            f"{row['activation_step']:>10.4f}{row['activation_absmax']:>10.2f}"
        )
    print("... worst 3 by weight SQNR:")
    for row in sorted(rows, key=lambda r: r["weight_sqnr_db"])[:3]:
        print(
            f"{row['layer']:<44}{row['activation_clipped_frac']*100:>8.2f}"
            f"{row['activation_sqnr_db']:>10.2f}{row['weight_sqnr_db']:>10.2f}"
        )
    out = probe.out or osp.join(
        args.logs_dir, f"quanterr_{probe.bits}bit.json"
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"bits": probe.bits, "rows": rows}, handle, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
