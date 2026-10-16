"""Separate AQCL entry point.

This file adds quantization-aware training on top of the unmodified CKAA
entry point in train_eval.py. CKAA experiments should continue to use
train_eval.py; only AQCL experiments should use this file.
"""

import argparse
import atexit
import faulthandler
import os
import os.path as osp
import sys
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

import train_eval as ckaa
import timm
from utils import misc
from utils.aqcl import AQCLContext
from utils.dataset_builder import define_dataset
from utils.logging import Logger
from utils.mod_adam import ModAdam


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--aqcl-enable", type=misc.str2bool, default=True)
    parser.add_argument("--aqcl-mode", choices=("fixed", "rpq", "rpq_saou"), default="rpq_saou")
    parser.add_argument("--aqcl-bits", type=int, default=8)
    parser.add_argument("--aqcl-low-bits", type=int, default=8)
    parser.add_argument("--aqcl-high-bits", type=int, default=16)
    parser.add_argument(
        "--aqcl-act-low-bits",
        type=int,
        default=0,
        help="Activation b_L. 0 = same as --aqcl-low-bits. The paper computes "
        "the weight sensitivity (Eq.14) and the activation sensitivity (Eq.15) "
        "separately and applies Eq.13 to each, so the two widths are "
        "independent. Decoupling them is useful here because the deployed size "
        "is driven by the weights while the accuracy loss is dominated by the "
        "activations.",
    )
    parser.add_argument(
        "--aqcl-act-high-bits",
        type=int,
        default=0,
        help="Activation b_H. 0 = same as --aqcl-high-bits.",
    )
    parser.add_argument(
        "--aqcl-act-bits",
        type=int,
        default=0,
        help="Activation bit-width in --aqcl-mode fixed. 0 = same as "
        "--aqcl-bits.",
    )
    parser.add_argument("--aqcl-lambda", type=float, default=10.0)
    parser.add_argument("--aqcl-alpha", type=float, default=5.0)
    parser.add_argument("--aqcl-theta", type=float, default=0.5)
    parser.add_argument(
        "--aqcl-allocation",
        choices=("mean", "paper", "budget"),
        default="mean",
        help="RPQ Eq.13 allocation rule. 'mean' is the original split at the "
        "mean sensitivity; 'paper' keeps only at-least-average layers at high "
        "precision; 'budget' promotes the most sensitive layers up to a fixed "
        "average bit-width budget.",
    )
    parser.add_argument("--aqcl-fisher-batches", type=int, default=8)
    parser.add_argument(
        "--aqcl-fisher-temperature",
        type=float,
        default=64.0,
        help="Temperature applied to the logits inside the Fisher/sensitivity "
        "pass only. Saturated logits produce an exactly-zero cross-entropy "
        "gradient; scaling restores a usable Fisher trace without affecting "
        "training (this pass never updates parameters).",
    )
    parser.add_argument(
        "--aqcl-backbone-bits",
        type=int,
        default=-1,
        help="Precision floor (bits) for frozen ViT backbone layers: "
        "0 = no floor (Eq.13 over the whole network), N = absolute floor of N "
        "bits, -1 (default) = floor = b_H. With -1 a 4/8 run keeps the backbone "
        "at 8-bit while a 2/4 run drops it to 4-bit so the deployed model "
        "actually shrinks (86 MB -> 44 MB); the frozen ViT holds ~86 of ~87 M "
        "parameters, so a fixed 8-bit backbone would otherwise dominate the size "
        "at every nominal bit-width.",
    )
    parser.add_argument("--aqcl-warmup-epochs", type=int, default=1)
    parser.add_argument(
        "--aqcl-post-task-fisher",
        type=misc.str2bool,
        default=False,
        help="Also refine the Fisher sensitivity from the post-task model. "
        "Disabled by default: saturated logits make those gradients zero on "
        "this dataset, which would overwrite the informative estimate.",
    )
    parser.add_argument(
        "--aqcl-per-token-activation",
        type=misc.str2bool,
        default=False,
        help="Give every token its own activation quantization range instead of "
        "one range per tensor. Reduces the activation quantization step for "
        "typical tokens when a few tokens carry large outliers.",
    )
    parser.add_argument(
        "--aqcl-range-tracking",
        choices=("frozen", "ema", "dynamic"),
        default="frozen",
        help="How the activation range behaves after calibration. 'frozen' "
        "keeps the calibrated grid (default); 'ema' keeps updating it so a long "
        "schedule can follow drifting activation statistics; 'dynamic' recomputes "
        "per batch (unstable, kept for ablation).",
    )
    parser.add_argument(
        "--aqcl-hadamard",
        type=misc.str2bool,
        default=False,
        help="Quantize in a Hadamard-rotated basis: the layer uses H_out W H_in "
        "and x H_in. With H H^T = I the FP32 function is unchanged, but the "
        "rotated weight has no isolated outlier coordinates, so a 2-4 bit grid "
        "represents it much better. Costs no extra model size.",
    )
    parser.add_argument(
        "--aqcl-hadamard-scope",
        choices=("attention", "attn_fc2", "no_fc1", "backbone"),
        default="attention",
        help="Which frozen layers get the Hadamard-rotated basis. 'attention' "
        "(default) restricts it to the attention qkv/proj projections, which "
        "operate on LayerNorm-bounded inputs. 'attn_fc2' adds mlp.fc2 (post-GELU "
        "input), 'no_fc1' adds everything except mlp.fc1, and 'backbone' rotates "
        "every frozen Linear, which was measured to diverge (loss=nan from epoch "
        "3) because rotating mlp.fc1 makes the pre-GELU feature dense and its "
        "activation ranges compound until fp16 overflows.",
    )
    parser.add_argument(
        "--aqcl-learnable-scale",
        type=misc.str2bool,
        default=False,
        help="Enable an LSQ-style learnable per-channel step multiplier on every "
        "quantizer. The calibrated range fixes the grid centre and the learned "
        "factor stretches or shrinks the step, i.e. it learns the clipping "
        "threshold at the same bit-width. Costs one scalar per quantizer.",
    )
    parser.add_argument(
        "--aqcl-scale-lr",
        type=float,
        default=0.001,
        help="Learning rate for the learnable step multipliers.",
    )
    parser.add_argument(
        "--aqcl-clip-percentile",
        type=float,
        default=0.0,
        help="Outlier clipping for quantization ranges: keep this quantile of "
        "each channel's distribution instead of the min/max extremes, e.g. 0.999 "
        "clips the outermost 0.1%%. Defaults to 0 (off): the per-channel quantile "
        "is a sort costing ~52 ms per call versus ~0.2 ms for min/max, which made "
        "a 2-bit epoch take 108 s instead of 12 s. Enable it for stability "
        "experiments. The spike protection that IS on by default is free: "
        "_recover_degenerate_range (span-ratio re-seeding) plus "
        "_sanitize_observation (non-finite repair).",
    )
    parser.add_argument(
        "--aqcl-calibrate-batches",
        type=int,
        default=4,
        help="Number of batches used to calibrate and freeze the activation "
        "quantization ranges at the start of each task's quantized training. "
        "0 disables calibration (per-batch dynamic min/max, the old behaviour).",
    )
    parser.add_argument("--aqcl-report", type=misc.str2bool, default=True)
    parser.add_argument(
        "--aqcl-instrument",
        type=misc.str2bool,
        default=False,
        help="Emit Phase-1 instrumentation records (Fisher sensitivity, bit "
        "allocation, Uo/Up dimensions, eigenvalue range, SAOU triggers, "
        "gradient cosine). Logging only; training behavior is unchanged.",
    )
    parser.add_argument(
        "--aqcl-instrument-verbose",
        type=misc.str2bool,
        default=True,
        help="Also print one line per layer to stdout/log.",
    )
    parser.add_argument(
        "--aqcl-instrument-json",
        type=str,
        default="",
        help="Path for the instrumentation JSON dump. Defaults to "
        "<logs-dir>/<logs-suffix>.instrument.json",
    )
    parser.add_argument(
        "--aqcl-debug-max-tasks",
        type=int,
        default=0,
        help="Debug only: stop after this many tasks (0 = run all tasks). "
        "Used to inspect the task-0 RPQ allocation of a multi-task run.",
    )
    aqcl_args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    ckaa_args = ckaa.get_args()
    for key, value in vars(aqcl_args).items():
        setattr(ckaa_args, key, value)
    # AQCL experiments follow the tabular CKAA baseline: the shared prompt is
    # fixed after the first task, and routing uses the selected task head.
    ckaa_args.freeze_shared_prompts_after_first = True
    ckaa_args.eval_local_task_head_task_only = True
    ckaa.args = ckaa_args
    return ckaa_args


def make_optimizer(model, context, scale_lr: float = 0.001):
    """Build the optimizer class used by the CKAA training loop.

    The CKAA loop constructs the optimizer itself from the CKAA training string,
    so the learnable quantization steps are not registered. This subclass injects
    them once, on the first optimization step, and then behaves like ModAdam.
    """
    if context is None:
        return ModAdam

    class AQCLModAdam(ModAdam):
        _aqcl_context = context
        _aqcl_model = model
        _aqcl_scale_lr = float(scale_lr)
        _aqcl_injected = False
        _aqcl_nonfinite_steps = 0

        def step(self, closure=None):
            AQCLModAdam._aqcl_context.modulate_gradients(AQCLModAdam._aqcl_model)
            if not AQCLModAdam._aqcl_injected:
                count = inject_quantizer_parameters(
                    self, AQCLModAdam._aqcl_context, AQCLModAdam._aqcl_scale_lr
                )
                if count:
                    print(
                        f"[AQCL] added {count} learnable quantization step "
                        f"parameters to the optimizer (lr={AQCLModAdam._aqcl_scale_lr})",
                        flush=True,
                    )
                AQCLModAdam._aqcl_injected = True
            # Low-bit training on this model can push activations to inf/nan; the
            # nan then propagates into every gradient and the run aborts with an
            # empty stderr (the original 2-bit failure). Drop only the offending
            # gradients and count the event, so the run survives and the log
            # records that it happened instead of dying silently.
            if os.environ.get("AQCL_GRAD_GUARD", "1") != "0":
                AQCLModAdam._aqcl_nonfinite_steps += _drop_nonfinite_gradients(self)
            return super().step(closure)

    return AQCLModAdam


def _drop_nonfinite_gradients(optimizer) -> int:
    """Zero any non-finite gradient in place; return how many parameters had one."""
    affected = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            grad = parameter.grad
            if grad is None:
                continue
            if not bool(torch.isfinite(grad).all()):
                affected += 1
                grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if affected:
        print(
            f"[AQCL] dropped non-finite gradients on {affected} parameter tensor(s) "
            f"this step (training continues; see AQCL_GRAD_GUARD)",
            flush=True,
        )
    return affected


def inject_quantizer_parameters(optimizer, context, lr: float) -> int:
    """Add the learnable quantization steps to the optimizer.

    The CKAA optimizer is built from the CKAA training string, so parameters
    outside that set are never registered. The quantization step multipliers are
    a separate, tiny parameter group; without this they would receive gradients
    but never be updated.
    """
    if optimizer is None or context is None:
        return 0
    params = context.quantizer_parameters()
    if not params:
        return 0
    optimizer.add_param_group({"params": params, "lr": float(lr)})
    return len(params)


def build_model(args, GVM, current_task_classes):
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
    if args.dataset in ("tabular", "fault_csv"):
        model = ckaa.convert_vit_to_tabular(
            model,
            signal_length=args.tabular_window_length,
            patch_size=args.tabular_patch_size,
            in_chans=args.tabular_in_chans,
        )
    context = AQCLContext(model, args) if args.aqcl_enable else None
    model = nn.DataParallel(model, device_ids=args.device_ids)
    return model, context


def _diagnose_logit_scale(GVM, taskid, curr_epoch, dataloader, model, context):
    """Logging-only: compare the quantized and FP32 forward passes.

    Extreme logit magnitudes make CrossEntropyLoss and its gradient collapse to
    exactly zero, which silently destroys the RPQ/SAOU Fisher estimate. This
    helper only reads the model and temporarily toggles quantizer bit-width
    buffers; it restores the previous allocation and never touches weights,
    gradients or optimizer state.
    """
    try:
        images, target, _ = next(iter(dataloader))
    except StopIteration:
        return
    device = model.module.device
    was_training = model.training
    previous_enabled = context.quantization_enabled
    # `set_quantization_enabled` re-derives bit-widths from the active
    # allocation map, so snapshot them and restore the exact values afterwards
    # to guarantee the diagnostic cannot perturb training.
    previous_weight_bits = {
        name: module.weight_quant.num_bits
        for name, module in context.targets.items()
    }
    previous_activation_bits = {
        name: module.activation_quant.num_bits
        for name, module in context.targets.items()
    }
    model.eval()

    def _forward_stats():
        with torch.no_grad():
            _, logits, _ = model(
                images.to(device, non_blocking=True),
                mode=GVM.args.train_tool + "_train",
                taskid=taskid,
                max_taskid=taskid,
            )
            logits = logits.detach().float()
            probabilities = logits.softmax(dim=1)
            return (
                float(logits.abs().max().item()),
                float(logits.std().item()),
                float(probabilities.min().item()),
                float(
                    (probabilities.max(dim=1).values >= 1.0 - 1e-6)
                    .float()
                    .mean()
                    .item()
                ),
            )

    try:
        # Quantized forward (current state).
        if context.quantization_enabled:
            ctx_absmax, ctx_std, ctx_minp, ctx_sat = _forward_stats()
            # Activation min/max actually observed by each quantizer.
            ranges = []
            for name, module in context.targets.items():
                if not name.endswith(("attn.qkv", "mlp.fc1", "mlp.fc2", "attn.proj")):
                    continue
                buffers = dict(module.activation_quant.named_buffers())
                if "last_x_min" in buffers:
                    ranges.append(
                        (
                            name,
                            float(buffers["last_x_min"].item()),
                            float(buffers["last_x_max"].item()),
                            float(buffers["last_step"].item()),
                        )
                    )
            top_ranges = sorted(
                ranges, key=lambda r: abs(r[2] - r[1]), reverse=True
            )[:5]
        else:
            ctx_absmax = ctx_std = ctx_minp = ctx_sat = float("nan")
            top_ranges = []

        # FP32 reference at the identical weights.
        context.set_quantization_enabled(False)
        fp32_absmax, fp32_std, fp32_minp, fp32_sat = _forward_stats()

        print(
            f"[AQCL-INSTRUMENT][logits] task={taskid} epoch={curr_epoch} "
            f"quant_enabled={previous_enabled} "
            f"quant_absmax={ctx_absmax:.3f} quant_saturated={ctx_sat:.3f} "
            f"fp32_absmax={fp32_absmax:.3f} fp32_saturated={fp32_sat:.3f} "
            f"amplification="
            f"{(ctx_absmax / max(fp32_absmax, 1e-12)) if previous_enabled else float('nan'):.2f}x",
            flush=True,
        )
        for name, x_min, x_max, step in top_ranges:
            print(
                f"[AQCL-INSTRUMENT][act_range] task={taskid} epoch={curr_epoch} "
                f"{name} min={x_min:.4f} max={x_max:.4f} step={step:.6f}",
                flush=True,
            )
    finally:
        context.quantization_enabled = previous_enabled
        for name, module in context.targets.items():
            module.weight_quant.set_bits(previous_weight_bits.get(name, 32))
            module.activation_quant.set_bits(
                previous_activation_bits.get(name, 32)
            )
        model.train(was_training)


def main():
    args = parse_args()
    ckaa.seed_etc_options(args.seed)
    logs_path = osp.join(args.logs_dir, args.logs_suffix + ".txt")
    os.makedirs(args.logs_dir, exist_ok=True)
    # Low-bit runs abort intermittently with a clean exit code and an empty
    # stderr, because stdout has already been redirected into the log file and
    # whatever raised never reached stderr. faulthandler writes to its own file
    # descriptor, so the faulting stack survives the redirect. Set
    # AQCL_FAULTHANDLER=0 to disable.
    if os.environ.get("AQCL_FAULTHANDLER", "1") != "0":
        fault_path = osp.join(args.logs_dir, args.logs_suffix + ".faulthandler.txt")
        faulthandler.enable(open(fault_path, "w"))

    # Exception hooks writing to their own unbuffered file descriptor, installed
    # *before* stdout is redirected. An uncaught exception after
    # ``sys.stdout = Logger(...)`` otherwise loses its traceback, which is exactly
    # how the 2-bit abort produced an exit code of 1 with empty stdout and stderr.
    # These hooks survive the redirect and do not depend on ``atexit`` firing.
    crash_handle = open(
        osp.join(args.logs_dir, args.logs_suffix + ".crash.txt"), "w", buffering=1
    )
    import threading
    import traceback as _traceback

    def _excepthook(exc_type, exc_value, exc_tb):
        crash_handle.write("uncaught exception in main thread\n")
        _traceback.print_exception(exc_type, exc_value, exc_tb, file=crash_handle)
        crash_handle.flush()
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _thread_excepthook(hook_args):
        crash_handle.write(
            f"uncaught exception in thread {hook_args.thread.name}\n"
        )
        _traceback.print_exception(
            hook_args.exc_type,
            hook_args.exc_value,
            hook_args.exc_traceback,
            file=crash_handle,
        )
        crash_handle.flush()

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook

    if os.environ.get("AQCL_DUMP_TRACEBACK", "0") == "1":

        def _dump_all_threads():
            crash_handle.write("\n=== stacks at interpreter exit ===\n")
            for thread_id, frame in sys._current_frames().items():
                crash_handle.write(f"\n--- thread {thread_id} ---\n")
                _traceback.print_stack(frame, file=crash_handle)
            crash_handle.flush()

        atexit.register(_dump_all_threads)
    sys.stdout = Logger(logs_path)

    instrumentation_json = args.aqcl_instrument_json or osp.join(
        args.logs_dir, args.logs_suffix + ".instrument.json"
    )
    context_holder = {"context": None}

    def _dump_instrumentation():
        context = context_holder["context"]
        if context is not None and getattr(args, "aqcl_instrument", False):
            try:
                context.instrumentation.dump(instrumentation_json)
            except Exception as error:  # pragma: no cover - diagnostics only
                print(f"[AQCL-INSTRUMENT] dump failed: {error}")

    atexit.register(_dump_instrumentation)

    GVM = ckaa.GlobalVarsManager()
    GVM.init_from_args(args)
    GVM.cache_dict["exp_start_time"] = ckaa.ttime()
    print("***** Start AQCL training *****")

    context = None
    model = None
    calibration_state = {"done": False}
    sensitivity_state = {"done": False}
    original_train_one_epoch = ckaa.train_one_epoch

    def train_one_epoch_with_warmup(
        GVM_,
        taskid_,
        curr_epoch,
        dataloader_,
        model_,
        criterion_,
        optimizer_,
        *args_,
        **kwargs_,
    ):
        if context is not None:
            warmup_epochs = max(int(args.aqcl_warmup_epochs), 0)
            enable_quantization = curr_epoch > warmup_epochs
            # 1) Collect the Eq.11/12 Fisher sensitivity from the warm-up model.
            #    Collecting it after quantized training collapses to zero on this
            #    dataset because the logits saturate and CE gradients vanish.
            #    Task 0 is measured on the pre-training model; later tasks are
            #    measured at the start of their quantized phase.
            if enable_quantization and not sensitivity_state["done"]:
                if curr_epoch == warmup_epochs + 1:
                    batches = int(getattr(args, "aqcl_fisher_batches", 0))
                    if batches > 0:
                        stats = context.collect_sensitivity(
                            model_.module
                            if isinstance(model_, nn.DataParallel)
                            else model_,
                            dataloader_,
                            taskid_,
                            criterion_,
                            max_batches=batches,
                            ema=1.0,
                            reset=(taskid_ != 0),
                        )
                        print(
                            f"[AQCL] sensitivity source="
                            f"{'pre-training' if taskid_ == 0 else 'task-start'}: "
                            f"{stats['nonzero_layers']}/{stats['num_layers']} layers "
                            f"non-zero from {stats['num_batches']} batch(es) "
                            f"(task {taskid_})",
                            flush=True,
                        )
                    sensitivity_state["done"] = True
            # 2) Calibrate and freeze the quantization ranges once per task.
            if (
                enable_quantization
                and not calibration_state["done"]
                and int(getattr(args, "aqcl_calibrate_batches", 0)) > 0
            ):
                stats = context.calibrate_ranges(
                    model_.module if isinstance(model_, nn.DataParallel) else model_,
                    dataloader_,
                    taskid_,
                    max_batches=int(args.aqcl_calibrate_batches),
                )
                calibration_state["done"] = True
                print(
                    f"[AQCL] calibrated quantization ranges from "
                    f"{stats['num_batches']} batch(es) over "
                    f"{stats['num_layers']} layers (task {taskid_}) "
                    f"[tracking={args.aqcl_range_tracking}]",
                    flush=True,
                )
                # Continue tracking the activation range after calibration when
                # requested: a long schedule needs the activation grid to follow
                # the drifting statistics, which a frozen grid cannot do.
                if args.aqcl_range_tracking != "frozen":
                    context.refresh_activation_ranges(args.aqcl_range_tracking)
            context.set_quantization_enabled(enable_quantization)
        if getattr(args, "aqcl_instrument", False) and context is not None:
            _diagnose_logit_scale(
                GVM_, taskid_, curr_epoch, dataloader_, model_, context
            )
        return original_train_one_epoch(
            GVM_,
            taskid_,
            curr_epoch,
            dataloader_,
            model_,
            criterion_,
            optimizer_,
            *args_,
            **kwargs_,
        )

    ckaa.train_one_epoch = train_one_epoch_with_warmup

    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#' * 30} Task: [{taskid + 1}/{GVM.cl_mngr.num_tasks}] {'#' * 30}")
        print(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        if not args.consecutive_training or taskid == 0:
            ckaa.current_task_classes = current_task_classes
            model, context = build_model(args, GVM, current_task_classes)
            context_holder["context"] = context
            GVM.cache_dict["pretrained_cfg"] = deepcopy(model.module.pretrained_cfg)
            GVM.cache_dict["aqcl"] = context
            ckaa.ModAdam = make_optimizer(model, context, args.aqcl_scale_lr)

        if args.consecutive_training and taskid > 0:
            ckaa.ModAdam = make_optimizer(model, context, args.aqcl_scale_lr)

        not_pretrained = ckaa.find_not_pretrained_params(
            model.module,
            pretrained_cfg=model.module.pretrained_cfg,
        )
        GVM.cache_dict["not_pretrained_params"] = not_pretrained
        GVM.update_label_maps(taskid, current_task_classes)
        GVM.cache_dict["training_string"] = args.training_string
        misc.check_param_training(not_pretrained, args.training_string)

        original_train_one_task = ckaa.train_one_task
        original_modify_head = ckaa.modify_head

        def modify_head_with_aqcl(*args, **kwargs):
            """Rebuild the head with CKAA, then re-register it for AQCL.

            ``modify_head`` grows ``model.head`` at every task, producing a new
            ``nn.Linear`` outside ``AQCLContext.targets``. Re-registering keeps
            the head quantized and visible to RPQ/SAOU, as the paper assumes for
            all trainable layers.
            """
            model_ = original_modify_head(*args, **kwargs)
            training_ = bool(kwargs.get("training", False)) or bool(
                len(args) > 2 and args[2]
            )
            if context is not None and training_:
                core = model_.module if isinstance(model_, nn.DataParallel) else model_
                registered = context.sync_classifier_head(core)
                if registered is not None:
                    gvm_ = kwargs.get("GVM", args[0] if args else None)
                    current = getattr(
                        getattr(gvm_, "cl_mngr", None), "current_taskid", "?"
                    )
                    print(
                        f"[AQCL] classifier head re-registered as a quantized "
                        f"target (task {current})",
                        flush=True,
                    )
            return model_

        ckaa.modify_head = modify_head_with_aqcl

        def train_one_task_with_aqcl(GVM_, taskid_, task_classes_, model_, **kwargs):
            if context is not None:
                context.before_task(taskid_)
                # Range calibration and Fisher collection must be redone for
                # every task: the class distribution changes.
                calibration_state["done"] = False
                sensitivity_state["done"] = False
            result = original_train_one_task(
                GVM_,
                taskid_,
                task_classes_,
                model_,
                **kwargs,
            )
            if context is not None:
                dataset = define_dataset(
                    GVM_,
                    task_classes_,
                    training=True,
                    transform_type=args.transform_type,
                    target_map_to_local=args.seperate_head,
                    expand_times=1,
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                )
                criterion = nn.CrossEntropyLoss().to(model_.module.device)
                context.set_quantization_enabled(True)
                context.after_task(
                    model_.module if isinstance(model_, nn.DataParallel) else model_,
                    dataloader,
                    taskid_,
                    criterion,
                    update_sensitivity=bool(
                        getattr(args, "aqcl_post_task_fisher", False)
                    ),
                )
                if args.aqcl_report:
                    sample = dataset[0][0].unsqueeze(0).to(model_.module.device)
                    report = context.inference_report(model_.module, sample)
                    print(
                        "AQCL deployment: "
                        f"size={report['model_size_mb']:.4f}MB, "
                        f"GFLOPs={report['inference_gflops']:.4f}, "
                        f"GBOPS={report['inference_gbops']:.4f}, "
                        f"extra_memory={report['inference_extra_memory_mb']:.4f}MB, "
                        f"weight_bits={report['weight_bits']}, "
                        f"activation_bits={report['activation_bits']}"
                    )
                if getattr(args, "aqcl_instrument", False):
                    # Phase 1: logging only. Emit per-layer Fisher sensitivity,
                    # bit allocation, subspace dimensions and gradient cosines.
                    context.capture_task_instrumentation(taskid_)
            return result

        if args.evaluation:
            if taskid == GVM.cl_mngr.num_tasks - 1:
                ckaa.evaluate_tasks_sofar(
                    GVM,
                    taskid,
                    model,
                    pretrained_model_path=osp.join(
                        "specific-shared",
                        "logs",
                        args.dataset,
                        args.evaluation_model_name + ".pkl",
                    ),
                )
                ckaa.task_ending_info(GVM)
        else:
            ckaa.train_one_task = train_one_task_with_aqcl
            model = train_one_task_with_aqcl(
                GVM,
                taskid,
                current_task_classes,
                model,
            )
            ckaa.train_one_task = original_train_one_task
            ckaa.modify_head = original_modify_head
            ckaa.evaluate_tasks_sofar(GVM, taskid, model)
            ckaa.task_ending_info(GVM)

        if getattr(args, "aqcl_debug_max_tasks", 0) and taskid + 1 >= int(
            args.aqcl_debug_max_tasks
        ):
            print(
                f"[AQCL-DEBUG] stopping after task {taskid + 1} "
                f"(--aqcl-debug-max-tasks {args.aqcl_debug_max_tasks})"
            )
            break

        if args.save_model:
            save_path = osp.join(
                "specific-shared",
                "logs",
                args.dataset,
                args.save_model_name + ".pkl",
            )
            os.makedirs(osp.dirname(save_path), exist_ok=True)
            # Deployment checkpoint: only inference-relevant tensors. Fisher,
            # covariance, SVD subspaces and gradient-modulation state are
            # deliberately excluded (they live in the AQCLContext only).
            torch.save(model.state_dict(), save_path)

    if getattr(args, "aqcl_instrument", False) and context is not None:
        json_path = args.aqcl_instrument_json or osp.join(
            args.logs_dir, args.logs_suffix + ".instrument.json"
        )
        context.instrumentation.dump(json_path)


if __name__ == "__main__":
    main()

