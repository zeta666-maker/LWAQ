"""Isolate why the AQCL Fisher pass produces all-zero gradients.

Replicates `AQCLContext._fisher_and_covariance` step by step on one batch and
prints, per targeted layer:
  * requires_grad state after forcing it True
  * whether `weight.grad` is None after loss.backward()
  * gradient norm and nonzero fraction
  * the same for a plain `nn.Linear` control layer

Diagnostic only; never used by training.
"""

from __future__ import annotations

import argparse
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
from utils.aqcl import QATLinear, convert_to_qat  # noqa: E402


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-device", type=str, default="cuda")
    parser.add_argument(
        "--probe-cpu",
        action="store_true",
        help="Force CPU execution and disable autodetected CUDA-dependent paths.",
    )
    known, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    args = ckaa.get_args()
    ckaa.args = args
    ckaa.seed_etc_options(args.seed)

    import timm
    from utils import misc

    GVM = ckaa.GlobalVarsManager()
    GVM.init_from_args(args)
    taskid, current_task_classes = next(iter(GVM.cl_mngr))
    GVM.update_label_maps(taskid, current_task_classes)
    ckaa.current_task_classes = current_task_classes
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
    device = torch.device(known.probe_device if torch.cuda.is_available() else "cpu")
    if known.probe_cpu:
        device = torch.device("cpu")
        # `train_one_epoch`-style helpers hard-code `.cuda()`; keep the probe on
        # CPU by routing every `.cuda()` call to the CPU device.
        original_cuda = torch.Tensor.cuda

        def _cpu_cuda(self, *a, **k):
            return self

        torch.Tensor.cuda = _cpu_cuda
        original_module_cuda = nn.Module.cuda

        def _module_cpu_cuda(self, *a, **k):
            return self

        nn.Module.cuda = _module_cpu_cuda
        args.use_amp = False
        print(f"[probe] forced CPU mode (patch cuda={original_cuda is not None})")
    model = model.to(device)
    # CKAA helpers index `model.module`; mirror that interface without
    # DataParallel when running the CPU-only variant.
    if device.type == "cpu":

        class _CpuModuleView(nn.Module):
            """Exposes `.module` without creating a module recursion cycle."""

            def __init__(self, inner):
                super().__init__()
                object.__setattr__(self, "_inner", inner)

            def __getattr__(self, item):
                if item == "module":
                    return object.__getattribute__(self, "_inner")
                return super().__getattr__(item)

            def forward(self, *a, **k):
                return object.__getattribute__(self, "_inner")(*a, **k)

        model = _CpuModuleView(model)
    else:
        model = nn.DataParallel(model, device_ids=[device.index or 0])
        model.to(device)
    model = ckaa.set_model_mode(
        GVM, taskid, model, training=True, training_string=args.training_string
    )
    model = ckaa.modify_head(GVM, model, training=True, task_classes=current_task_classes)
    model.train()

    targets = convert_to_qat(model.module, low_bits=8)
    print(f"[probe] quantized targets: {len(targets)}")

    for name, module in targets.items():
        module.weight.requires_grad_(True)

    requires_grad_true = [
        name for name, module in targets.items() if module.weight.requires_grad
    ]
    print(f"[probe] requires_grad=True after forcing: {len(requires_grad_true)}/{len(targets)}")

    dataset = ckaa.define_dataset(
        GVM,
        current_task_classes,
        training=True,
        transform_type=args.transform_type,
        target_map_to_local=args.seperate_head,
        expand_times=1,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    criterion = nn.CrossEntropyLoss().to(device)

    grad_seen: dict[str, dict] = {}
    activation_hook_calls = {"count": 0, "grad_input_none": 0}

    def activation_hook(name):
        def hook(_module, grad_input, _grad_output):
            activation_hook_calls["count"] += 1
            if not grad_input or grad_input[0] is None:
                activation_hook_calls["grad_input_none"] += 1
        return hook

    handles = [
        module.register_full_backward_hook(activation_hook(name))
        for name, module in targets.items()
    ]

    for batch_index, (images, target, _) in enumerate(dataloader):
        if batch_index >= 1:
            break
        images = images.to(device)
        target = target.to(device)
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(args.use_amp) and device.type == "cuda",
        ):
            _, logits, _ = model(
                images,
                mode=args.train_tool + "_train",
                taskid=taskid,
                max_taskid=taskid,
            )
            loss = criterion(logits, target)
        print(f"[probe] loss={float(loss.detach().item()):.6f} logits={tuple(logits.shape)}")
        loss.backward()

        for name, module in targets.items():
            grad = module.weight.grad
            if grad is None:
                grad_seen[name] = {"grad": "None"}
                continue
            grad_f = grad.detach().float()
            grad_seen[name] = {
                "norm": float(grad_f.norm().item()),
                "absmax": float(grad_f.abs().max().item()),
                "nonzero_frac": float((grad_f != 0).float().mean().item()),
                "numel": int(grad_f.numel()),
            }

    for handle in handles:
        handle.remove()

    print(f"[probe] backward hooks fired: {activation_hook_calls}")

    ordered = sorted(
        grad_seen.items(),
        key=lambda kv: kv[1].get("norm", -1.0),
        reverse=True,
    )
    print("[probe] top-10 layers by gradient norm:")
    for name, info in ordered[:10]:
        print(f"   {name}: {info}")
    print("[probe] bottom-5 layers by gradient norm:")
    for name, info in ordered[-5:]:
        print(f"   {name}: {info}")

    none_count = sum(1 for info in grad_seen.values() if info.get("grad") == "None")
    zero_count = sum(
        1 for info in grad_seen.values() if info.get("norm", 1.0) == 0.0
    )
    print(
        f"[probe] summary: layers={len(grad_seen)} grad_None={none_count} "
        f"grad_zero={zero_count}"
    )

    # Control: confirm that requires_grad toggling alone suffices.
    control = nn.Linear(8, 4).to(device)
    control.weight.requires_grad_(False)
    x = torch.randn(2, 8, device=device)
    out = control(x).sum()
    out.backward()
    print(
        "[probe] control (requires_grad False before forward): "
        f"grad is None -> {control.weight.grad is None}"
    )
    control.weight.requires_grad_(True)
    control.zero_grad(set_to_none=True)
    out = control(x).sum()
    out.backward()
    grad = control.weight.grad
    print(
        "[probe] control (requires_grad True before forward): "
        f"grad None -> {grad is None}, norm -> "
        f"{None if grad is None else float(grad.norm().item()):.6f}"
    )


if __name__ == "__main__":
    main()
