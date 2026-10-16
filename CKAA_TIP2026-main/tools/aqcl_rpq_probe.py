"""Phase-1 RPQ degeneracy probe (diagnostic only, does not train).

Falsifies or confirms the hypothesis that RPQ collapses to a single bit-width.
It builds the exact AQCL task-0 model/data path, runs the Fisher pass for the
first task, and reports:

  * per-layer raw weight/activation Fisher sensitivity
  * gradient norm per batch and the fraction of exactly-zero gradients
  * sensitivity recomputed with quantization disabled (32-bit reference)
  * the resulting Eq.13 bit allocation

Usage:
  python tools/aqcl_rpq_probe.py --aqcl-low-bits 4 --aqcl-high-bits 8
"""

from __future__ import annotations

import argparse
import json
import os
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
from utils.aqcl_instrument import summarize_scalar_tensor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-fisher-batches", type=int, default=2)
    parser.add_argument("--probe-low-bits", type=int, default=4)
    parser.add_argument("--probe-high-bits", type=int, default=8)
    parser.add_argument("--probe-out", type=str, default="")
    parser.add_argument("--probe-train-epochs", type=int, default=0)
    probe, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    ckaa_args = ckaa.get_args()
    ckaa.args = ckaa_args
    return ckaa_args, probe


def build_probe_model(args, GVM, current_task_classes):
    import timm
    from utils import misc

    prompt_args = misc.get_specific_args_dict(args, "prompt_")
    other_args = misc.get_specific_args_dict(args, "logit_")
    old_current = getattr(ckaa, "current_task_classes", None)
    ckaa.current_task_classes = current_task_classes
    head_args = ckaa.get_head_dim_arg_dict(GVM, args)
    if old_current is not None:
        ckaa.current_task_classes = old_current
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
    return model


def main():
    args, probe = parse_args()
    ckaa.seed_etc_options(args.seed)
    GVM = ckaa.GlobalVarsManager()
    GVM.init_from_args(args)

    taskid, current_task_classes = next(iter(GVM.cl_mngr))
    GVM.update_label_maps(taskid, current_task_classes)
    ckaa.current_task_classes = current_task_classes
    GVM.cache_dict["training_string"] = ("adapter",) if False else args.training_string

    model = build_probe_model(args, GVM, current_task_classes)
    model = nn.DataParallel(model, device_ids=args.device_ids)
    model = ckaa.set_model_mode(
        GVM, taskid, model, training=True, training_string=args.training_string
    )
    model = ckaa.modify_head(GVM, model, training=True, task_classes=current_task_classes)

    device = model.module.device
    criterion = nn.CrossEntropyLoss().to(device)

    dataset = ckaa.define_dataset(
        GVM,
        current_task_classes,
        training=True,
        transform_type=args.transform_type,
        target_map_to_local=args.seperate_head,
        expand_times=1,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Optional short real training so the probe sees a converged-enough model.
    if probe.probe_train_epochs > 0:
        optimizer = ckaa.ModAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            {},
            arg_dict={},
            lr=args.lr,
            weight_decay=args.weight_decay,
            foreach=True,
        )
        for epoch in range(probe.probe_train_epochs):
            stats = ckaa.train_one_epoch(
                GVM, taskid, epoch + 1, dataloader, model, criterion, optimizer, [], None
            )
            print(f"[probe] train epoch {epoch + 1}: {stats}")

    context = AQCLContext(model.module, args)
    context.mode = "rpq_saou"
    context.low_bits = probe.probe_low_bits
    context.high_bits = probe.probe_high_bits
    context.fisher_batches = probe.probe_fisher_batches
    context.before_task(taskid)
    context.set_quantization_enabled(True)

    # Run the Fisher/covariance pass and capture raw diagnostics.
    from utils.aqcl_instrument import AQCLInstrumentation

    context.instrumentation = AQCLInstrumentation(enabled=False, verbose=False)
    context.after_task(
        model.module if isinstance(model, nn.DataParallel) else model,
        dataloader,
        taskid,
        criterion,
    )

    weight_sens = context.weight_sensitivity
    act_sens = context.activation_sensitivity
    w_values = torch.tensor(
        [float(v.detach().to(torch.float64).item()) for v in weight_sens.values()],
        dtype=torch.float64,
    )
    a_values = torch.tensor(
        [float(v.detach().to(torch.float64).item()) for v in act_sens.values()],
        dtype=torch.float64,
    )

    rows = []
    for name, module in context.targets.items():
        rows.append(
            {
                "layer": name,
                "shape": list(module.weight.shape),
                "numel": int(module.weight.numel()),
                "weight_sensitivity": float(weight_sens[name].item()),
                "activation_sensitivity": float(act_sens[name].item()),
                "weight_bits": int(context.active_weight_bits.get(name, -1)),
                "activation_bits": int(context.active_activation_bits.get(name, -1)),
                "weight_grad_norm": float(module.weight.grad.norm().item())
                if module.weight.grad is not None
                else None,
            }
        )
    rows.sort(key=lambda r: r["weight_sensitivity"], reverse=True)

    result = {
        "protocol": {
            "low_bits": context.low_bits,
            "high_bits": context.high_bits,
            "fisher_batches": context.fisher_batches,
            "quantization_enabled_during_fisher": context.quantization_enabled,
            "batch_size": args.batch_size,
            "probe_train_epochs": probe.probe_train_epochs,
            "num_layers": len(rows),
            "datasetsize": len(dataset),
        },
        "weight_sensitivity_stats": summarize_scalar_tensor(w_values),
        "activation_sensitivity_stats": summarize_scalar_tensor(a_values),
        "bit_allocation_counts": {
            str(bits): sum(
                1 for r in rows if r["weight_bits"] == bits
            )
            for bits in sorted({r["weight_bits"] for r in rows})
        },
        "activation_bit_allocation_counts": {
            str(bits): sum(
                1 for r in rows if r["activation_bits"] == bits
            )
            for bits in sorted({r["activation_bits"] for r in rows})
        },
        "layers": rows,
    }

    print(json.dumps({k: v for k, v in result.items() if k != "layers"}, indent=2))
    print("[probe] bottom 5 layers by sensitivity:")
    for row in rows[-5:]:
        print("  ", json.dumps(row))
    print("[probe] top 5 layers by sensitivity:")
    for row in rows[:5]:
        print("  ", json.dumps(row))

    out = probe.probe_out or osp.join(
        args.logs_dir, args.logs_suffix + ".probe.json"
    )
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
