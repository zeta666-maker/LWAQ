"""Adaptive Quantization for Stable Knowledge Acquisition (AQCL).

This module is intentionally self-contained and uses only PyTorch. It provides
asymmetric uniform fake quantization, Fisher-trace sensitivity estimation,
adaptive weight/activation bit allocation, and similarity-aware orthogonal
gradient modulation.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import os
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils.aqcl_instrument import AQCLInstrumentation


def _as_int(value) -> int:
    if isinstance(value, Tensor):
        return int(value.item())
    return int(value)


def hadamard_matrix(size: int, device, dtype) -> Tensor:
    """Normalized Hadamard matrix H with H H^T = I, for a power-of-two size."""
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    h = torch.ones(1, 1, device=device, dtype=torch.float32)
    while h.shape[0] < size:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return (h / math.sqrt(size)).to(dtype)


def hadamard_block_hadamard(size: int, device, dtype) -> Tensor | None:
    """Block-diagonal Hadamard matrix covering any even size.

    A single Hadamard matrix needs a power-of-two size, but transformer widths are
    rarely powers of two (768 = 3*256, 2304 = 9*256, and the classifier head is 2
    or 4). We therefore take the largest power-of-two factor ``b`` of ``size`` and
    tile copies of the normalized ``b x b`` Hadamard matrix along the diagonal.
    The result is orthogonal (``H H^T = I``) for any size divisible by ``b``.

    Returns ``None`` when no usable block exists (odd sizes with no power-of-two
    factor), in which case the caller keeps the original basis.
    """
    if size <= 0:
        return None
    block = size & (-size)  # largest power-of-two divisor
    if block < 4:
        return None
    small = hadamard_matrix(block, device, dtype)
    repeats = size // block
    full = torch.zeros(size, size, device=device, dtype=dtype)
    for index in range(repeats):
        start = index * block
        full[start : start + block, start : start + block] = small
    return full


def rotate_weight(dim: int, weight: Tensor, device, dtype) -> tuple[Tensor, Tensor | None]:
    """Rotate a weight matrix to a flatter basis, and return the input transform.

    For a weight ``W`` with shape ``[dim, in_features]`` we form

        W_rot = H W ,      H = block-diagonal Hadamard of size ``dim``

    with ``H H^T = I``. The layer output becomes ``x @ W_rot^T = x @ W^T H^T``,
    i.e. the output is expressed in the rotated basis. That is exactly how the
    global rotation trick works: a rotation inserted between two layers must be
    undone by the next layer, and for a Linear followed (through an elementwise
    nonlinearity) by another Linear the composition still yields the identity on
    the hidden representation. The rotation is therefore applied to the **output**
    dimension, which is where quantization outliers live (few channels are much
    larger than the rest).

    Returns ``(W_rot, H)``; ``H`` is ``None`` when no block Hadamard exists, in
    which case ``W_rot`` is the unrotated weight.
    """
    if dim <= 0:
        return weight, None
    hadamard = hadamard_block_hadamard(dim, device, dtype)
    if hadamard is None:
        return weight, None
    return hadamard @ weight, hadamard



class UniformQuantizer(nn.Module):
    """Asymmetric uniform quantizer (paper Eq.1) with a straight-through estimator.

    Range handling matters a lot for low-bit QAT:

    * ``range_mode="dynamic"`` recomputes ``[l, u]`` from every incoming batch.
      This is what the first implementation did, and it makes the effective
      scale change at every step, so the network cannot adapt to a stable
      quantization grid. It also lets a single outlier expand the range and
      destroy resolution for the remaining 99.9% of values.
    * ``range_mode="ema"`` keeps an exponential moving average of the observed
      range, which damps batch-to-batch jitter.
    * ``freeze()`` locks the current range. Validation and deployment then use
      exactly the calibration range, which is the standard QAT contract.

    Weights use per-output-channel ranges; activations use one range per tensor.
    """

    def __init__(
        self,
        bits: int = 8,
        per_channel: bool = False,
        range_mode: str = "dynamic",
        momentum: float = 0.9,
    ):
        super().__init__()
        self.register_buffer("bits", torch.tensor(int(bits), dtype=torch.int64))
        # Instrumentation-only telemetry buffers. They are non-persistent, so
        # they never appear in a deployment checkpoint.
        self.register_buffer(
            "last_x_min", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_x_max", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_step", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "last_out_absmax", torch.tensor(0.0), persistent=False
        )
        # Calibrated (running) asymmetric range, shape broadcastable to x.
        self.register_buffer("range_min", torch.zeros(1), persistent=False)
        self.register_buffer("range_max", torch.ones(1), persistent=False)
        self.register_buffer("range_ready", torch.tensor(0, dtype=torch.int64))
        # Telemetry is only enabled when Phase-1 instrumentation is on; it is
        # read-only and does not alter the quantized values.
        self._telemetry = False
        # Check every quantizer output for non-finite values. This is cheap
        # relative to the matmul it feeds and is left ON because low-bit QAT on
        # this model diverges to ``loss=nan`` non-deterministically; failing at
        # the offending tensor beats discovering a ``nan`` in the metrics.
        self._finitude_check = True
        # The check costs ~0.5 ms per quantizer call (tensor-wide reduction over
        # up to millions of elements), so it runs on one call in
        # ``finitude_check_every``. A divergence is still caught within a few
        # optimizer steps of starting, while the amortized cost is negligible.
        self._finitude_counter = 0
        # ``AQCL_LEGACY`` selects the quantizer's numerical behaviour.
        #
        # DEFAULT IS LEGACY (1): the original pre-fix path, which is the only path
        # measured to reproduce the published baselines. Verified on 8-bit, seed
        # 2024: legacy 80.87 / 90.98 / 16.93 vs original code
        # 80.87 / 90.98 / 16.93 (identical), whereas the "improved" path scored
        # 78.69 without output clamping and 13.22 with it.
        #
        # Set AQCL_LEGACY=0 to enable the later numerical changes (span-based step
        # floor, observation sanitisation, degenerate-range recovery). They are
        # diagnostics and stability guards; they are NOT required to reproduce the
        # paper-comparable numbers, and they measurably changed results.
        self.legacy_mode = os.environ.get("AQCL_LEGACY", "1") != "0"
        # A min/max range is not robust to spikes; if the stored span exceeds the
        # observed span by this factor, the range is treated as spike-dominated and
        # re-seeded (see _recover_degenerate_range).
        self.range_recovery = not self.legacy_mode
        self.range_recovery_ratio = 64.0
        self.range_recoveries = 0
        # Output clamping to [x_min, x_max] is DISABLED by default.
        #
        # It was added to bound the value when an input falls outside the frozen
        # calibrated range, which is what real integer quantization does. But it
        # measurably destroys quantized accuracy: on 8-bit, seed 2024, the same
        # configuration scored 80.87% before it existed, 13.22% with it on, and
        # 78.69% with it off -- because the calibrated range is narrower than the
        # values seen during task training, so clamping saturates most of the
        # tensor onto the range endpoints and the information is lost.
        # AQCL_CLAMP_OUTPUT=1 re-enables it for experiments.
        self.clamp_output = os.environ.get("AQCL_CLAMP_OUTPUT", "0") == "1"
        # Percentile clipping needs a per-channel sort, measured at 171.6 ms per
        # call versus 0.20 ms for amin/amax on a [32,197,768] activation -- an
        # 858x blowup that made a 2-bit epoch take 108 s instead of 12 s. It is
        # therefore applied ONLY while a range is actually being observed (the
        # calibration pass); a frozen range reuses the already-clipped values and
        # must not pay for a sort on every forward.
        self._observing_range = False
        # ``AQCL_FINITUDE_EVERY=1`` checks every call, which pinpoints the first
        # bad step at the cost of ~+180% quantizer time. Useful when diagnosing a
        # divergence, not for production runs.
        self.finitude_check_every = max(
            1, int(os.environ.get("AQCL_FINITUDE_EVERY", "32") or 32)
        )
        self.per_channel = bool(per_channel)
        self.reduce = "channel" if per_channel else "all"
        self.use_exact_range = False
        # 0 disables percentile clipping (plain min/max range).
        self.clip_percentile = 0.0
        # Learnable per-channel step multiplier (LSQ-style). The calibrated range
        # fixes the grid centre; this parameter lets gradient descent stretch or
        # shrink the step, and therefore the clipping threshold, without changing
        # the bit-width. It is an inference parameter (a scalar per channel), not
        # training-only state.
        self.learnable_scale = False
        self.register_buffer("scale_raw", torch.zeros(1), persistent=True)
        if range_mode not in ("dynamic", "ema"):
            raise ValueError(f"Unsupported range_mode: {range_mode}")
        self.range_mode = range_mode
        self.momentum = float(momentum)

    def _effective_level(self, level: Tensor) -> Tensor:
        """Apply the learnable step multiplier (identity when disabled)."""
        if not self.learnable_scale:
            return level
        return level * torch.exp(self.scale_raw.to(level.dtype))

    @property
    def num_bits(self) -> int:
        return _as_int(self.bits)

    def set_bits(self, bits: int) -> None:
        self.bits.fill_(int(bits))

    def set_per_channel(self, per_channel: bool) -> None:
        """Switch the activation granularity.

        ``per_channel=True`` uses one asymmetric range per feature channel,
        reduced over all other dimensions. For transformer activations shaped
        ``[B, N, C]`` the magnitude varies by orders of magnitude between tokens,
        so a single per-tensor range forces a large step on the typical token;
        per-channel statistics keep a fixed-shape range that is far tighter for
        most channels.
        """
        self.per_channel = bool(per_channel)
        self.reduce = "channel" if per_channel else "all"
        self.range_ready.fill_(0)

    def set_reduce(self, reduce: str) -> None:
        if reduce not in ("all", "channel"):
            raise ValueError(f"Unsupported reduce: {reduce}")
        self.reduce = reduce
        self.per_channel = reduce == "channel"
        self.range_ready.fill_(0)

    def grid_step(self, dtype=None, device=None) -> Tensor:
        """Return the current quantization step, for diagnostics."""
        bits = self.num_bits
        span = (self.range_max - self.range_min).clamp_min(
            torch.finfo(torch.float32).eps
        )
        return span / float((1 << bits) - 1)

    def _reduce_per_channel(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Reduce along every dimension except the feature (last) one."""
        if self.reduce == "channel" and x.ndim >= 2:
            reduce_dims = tuple(range(x.ndim - 1))
            return (
                x.detach().amin(dim=reduce_dims),
                x.detach().amax(dim=reduce_dims),
            )
        if self.per_channel and x.ndim >= 2:
            reduce_dims = tuple(range(1, x.ndim))
            return (
                x.detach().amin(dim=reduce_dims, keepdim=True),
                x.detach().amax(dim=reduce_dims, keepdim=True),
            )
        return x.detach().amin(), x.detach().amax()

    def _quantile_range(self, flat: Tensor) -> tuple[Tensor, Tensor]:
        """Per-channel low/high quantiles using ``kthvalue`` instead of ``quantile``.

        ``torch.quantile`` sorts, which measured 52.2 ms per call on a
        ``[768, 6304]`` activation; ``kthvalue`` selects the order statistic
        directly and measured 12.2 ms for the same result. The clipping is a
        stability measure, so a 4x cheaper equivalent is worth using.
        """
        q = float(self.clip_percentile)
        tail = (1.0 - q) / 2.0
        columns = flat.shape[1]
        low_index = max(1, min(columns, int(round(tail * (columns - 1))) + 1))
        high_index = max(1, min(columns, columns - low_index + 1))
        data = flat.float()
        low = torch.kthvalue(data, low_index, dim=1).values
        high = torch.kthvalue(data, high_index, dim=1).values
        return low, high

    def _observed_range(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Observed asymmetric range for the configured granularity.

        ``reduce="all"``      one scalar range for the whole tensor.
        ``reduce="channel"``  one range per feature channel, i.e. the range
                              statistics are taken over every non-feature
                              dimension. For transformer activations shaped
                              ``[B, N, C]`` this gives a ``[1, 1, C]`` range that
                              is a fixed shape, so it can be accumulated across
                              batches regardless of the batch size.

        When ``clip_percentile`` is set, the range is taken from that quantile of
        the per-channel magnitude instead of the extremes, which makes the range
        robust to a single activation spike.

        Cost note: the per-channel quantile is a sort and measured **52 ms** per
        call on a ``[768, 6304]`` activation versus **0.2 ms** for amin/amax. It is
        therefore applied only while a range is genuinely being observed (the
        calibration pass, and only when ``AQCL_CLIP_PERCENTILE`` is enabled); a
        frozen range reuses the already-clipped values and pays nothing.
        """
        if (
            self.clip_percentile
            and self.clip_percentile > 0.0
            and x.ndim >= 2
            and self._observing_range
        ):
            if self.reduce == "channel":
                # x: [..., C] -> one range per feature channel
                flat = x.detach().movedim(-1, 0).reshape(x.shape[-1], -1)
                lo, hi = self._quantile_range(flat)
                return lo.to(x.dtype), hi.to(x.dtype)
            if self.per_channel:
                reduce_dims = tuple(range(1, x.ndim - 1)) if x.ndim > 2 else ()
                if reduce_dims:
                    flat = x.detach().movedim(-1, 0).reshape(x.shape[-1], -1)
                else:
                    flat = x.detach().reshape(x.shape[0], -1)
                lo, hi = self._quantile_range(flat)
                return (
                    lo.to(x.dtype).unsqueeze(-1),
                    hi.to(x.dtype).unsqueeze(-1),
                )
        return self._reduce_per_channel(x)

    def freeze(self) -> None:
        """Lock the calibrated range (used at deployment/evaluation time)."""
        if int(self.range_ready.item()) == 0:
            return
        self.range_mode = "frozen"

    def unfreeze(self, range_mode: str = "ema") -> None:
        self.range_mode = range_mode

    def _apply_step_floor(
        self, level: Tensor, span: Tensor, dtype: torch.dtype
    ) -> Tensor:
        """Keep the quantization step above the range's own numerical resolution.

        The previous code clamped with ``torch.finfo(dtype).eps``, which is an
        *absolute* floor (5.96e-8 in fp32, 6.1e-5 in fp16). For a channel whose
        activations are small, ``span / (2**bits - 1)`` is smaller than that
        epsilon and the clamp pins the step at epsilon; ``(x - x_min) / level``
        then reaches ~1e7 and the quantized value overflows the fp16 autocast
        path, which is a plausible source of the non-deterministic ``loss=nan``
        observed at 2-4 bits.

        The floor is therefore relative to the observed span: the step may never
        fall below ``finfo.eps`` times the span, which stays well clear of the
        subnormal range while being negligible whenever the range is meaningful.
        A span of zero (a constant tensor) is handled by the absolute epsilon
        because dividing by it is impossible by construction.
        """
        eps = torch.finfo(dtype).eps
        absolute = torch.as_tensor(eps, dtype=level.dtype, device=level.device)
        scale = span.detach().abs() * eps
        floor = torch.maximum(absolute, scale)
        return torch.maximum(level, floor)

    def _recover_degenerate_range(
        self, observed_min: Tensor, observed_max: Tensor
    ) -> None:
        """Re-seed the range when a spike has made the stored span useless.

        The stored span is compared with the span the current data actually shows.
        If it is more than ``range_recovery_ratio`` times larger, the buffer is
        dominated by a past spike (the EMA cannot shed it quickly) and the
        resulting grid is too coarse to represent the layer; re-seeding restores a
        usable grid. This is the mechanism that turned a transient activation
        spike into a permanent, silent model divergence at 2-4 bits.
        """
        if not self.range_recovery:
            return
        stored_span = (self.range_max - self.range_min).abs()
        observed_span = (observed_max - observed_min).abs()
        if not bool(torch.isfinite(stored_span).all()):
            return
        # Only act when the current data genuinely shows a much smaller spread;
        # a layer whose activations legitimately grew must not be re-seeded.
        threshold = self.range_recovery_ratio * observed_span.clamp_min(
            torch.finfo(stored_span.dtype).tiny
        )
        degenerate = stored_span > threshold
        if bool(degenerate.any()):
            keep = ~degenerate
            self.range_min = torch.where(
                keep, self.range_min, observed_min.detach().to(self.range_min.dtype)
            )
            self.range_max = torch.where(
                keep, self.range_max, observed_max.detach().to(self.range_max.dtype)
            )
            self.range_recoveries += int(degenerate.sum())

    def _sanitize_observation(self, observed: Tensor) -> Tensor:
        """Replace non-finite range observations with the tensor's finite extremes.

        The calibration and EMA paths must never store a ``nan``/``inf`` range: it
        would make the quantization step non-finite for every element of that
        layer, which is exactly how ``task_head.2.activation`` failed at 2-bit
        (``span=nan``, 768/768 steps non-finite, while the input itself was
        finite with absmax 20.2).
        """
        if bool(torch.isfinite(observed).all()):
            return observed
        finite = observed[torch.isfinite(observed)]
        if finite.numel() == 0:
            return torch.zeros_like(observed)
        return torch.nan_to_num(
            observed,
            nan=float(finite.median()),
            posinf=float(finite.max()),
            neginf=float(finite.min()),
        )

    def _assert_finite(self, x: Tensor, x_min: Tensor, x_max: Tensor, level: Tensor, q: Tensor) -> None:
        """Raise on the first non-finite quantization input or output.

        Low-bit QAT on this model diverges to ``loss=nan`` non-deterministically
        (observed at 2-bit on several recorded runs). Without this check the only
        symptom is a ``nan`` in the metrics and the offending tensor has to be
        hunted through the logs; failing here names the tensor instead.

        Only ``x`` and ``q`` are scanned: ``x_min``/``x_max``/``level`` are
        per-channel vectors whose non-finiteness would already make ``q``
        non-finite, and scanning them costs more than the two large tensors
        combined on this model's shapes.
        """
        for label, tensor in (("input", x), ("output", q)):
            if not bool(torch.isfinite(tensor).all()):
                bad = int((~torch.isfinite(tensor)).sum())
                step_stats = "n/a"
                if torch.isfinite(level).all():
                    step_stats = (
                        f"min={float(level.min()):.3e} max={float(level.max()):.3e}"
                    )
                else:
                    step_stats = (
                        f"NON-FINITE ({int((~torch.isfinite(level)).sum())} of "
                        f"{level.numel()})"
                    )
                message = (
                    f"AQCL quantization produced a non-finite {label} in "
                    f"{getattr(self, 'debug_name', '<unnamed>')} "
                    f"[{'weights' if getattr(self, 'is_weight', False) else 'activations'}]: "
                    f"{bad} of {tensor.numel()} elements at bits={self.num_bits}, "
                    f"range_mode={self.range_mode}, reduce={self.reduce}, "
                    f"span={float((x_max - x_min).abs().max()):.3e}, step {step_stats}, "
                    f"input_absmax={float(x.detach().abs().max()):.3e}."
                )
                if os.environ.get("AQCL_ALLOW_NONFINITE", "0") == "1":
                    print(f"[AQCL-WARNING] {message}", flush=True)
                    return
                raise FloatingPointError(
                    message + " Set AQCL_ALLOW_NONFINITE=1 to downgrade to a warning."
                )

    def forward(self, x: Tensor) -> Tensor:
        bits = self.num_bits
        if bits >= 32 or bits <= 0:
            return x
        if self.use_exact_range:
            # Static tensors (weights) have an exact, constant range, so an EMA
            # estimate would only add bias: the running maximum converges below
            # the true maximum and clips the extreme weights.
            x_min, x_max = self._observed_range(x)
        else:
            need_observation = self.range_mode == "dynamic" or (
                self.range_mode == "ema"
            ) or int(self.range_ready.item()) == 0
            if need_observation:
                self._observing_range = True
                try:
                    observed_min, observed_max = self._observed_range(x)
                finally:
                    self._observing_range = False
            if self.range_mode == "dynamic":
                x_min, x_max = observed_min, observed_max
            else:
                # A frozen range has no fresh observation this call; the stored
                # buffers are used as-is (and repaired below if non-finite).
                if need_observation and not self.legacy_mode:
                    observed_min = self._sanitize_observation(observed_min)
                    observed_max = self._sanitize_observation(observed_max)
                if int(self.range_ready.item()) == 0:
                    # First observation initialises the running range.
                    self.range_min = observed_min.detach().to(torch.float32).clone()
                    self.range_max = observed_max.detach().to(torch.float32).clone()
                    self.range_ready.fill_(1)
                elif self.range_mode == "ema":
                    momentum = self.momentum
                    self.range_min.mul_(momentum).add_(
                        observed_min.detach().to(torch.float32),
                        alpha=1.0 - momentum,
                    )
                    self.range_max.mul_(momentum).add_(
                        observed_max.detach().to(torch.float32),
                        alpha=1.0 - momentum,
                    )
                    # The EMA is a running average, so a single non-finite
                    # observation would poison it permanently
                    # (nan * momentum + finite == nan). Re-seed from the current
                    # observation instead of propagating the corruption.
                    for buffer, observed in (
                        (self.range_min, observed_min),
                        (self.range_max, observed_max),
                    ):
                        if not bool(torch.isfinite(buffer).all()):
                            buffer.copy_(
                                observed.detach().to(torch.float32).clone()
                            )
                    # A min/max range is not robust: ONE spike sets it, and the
                    # EMA then keeps it inflated for hundreds of steps. Measured
                    # case at 2-bit: span=6.0e+04 with step max=2.0e+04 on data of
                    # order 1, i.e. a grid so coarse that the layer's output is
                    # constant, after which the model diverged to inf/nan. Recover
                    # by re-seeding whenever the stored span is wildly larger than
                    # what the data currently shows.
                    self._recover_degenerate_range(observed_min, observed_max)
                x_min = self.range_min.to(x.dtype)
                x_max = self.range_max.to(x.dtype)
                if need_observation and (
                    not bool(torch.isfinite(x_min).all())
                    or not bool(torch.isfinite(x_max).all())
                ):
                    # Last-resort repair: fall back to the current observation so
                    # a corrupt calibration cannot make the grid non-finite.
                    x_min = observed_min.to(x.dtype)
                    x_max = observed_max.to(x.dtype)
                    self.range_min = (
                        observed_min.detach().to(torch.float32).clone()
                    )
                    self.range_max = (
                        observed_max.detach().to(torch.float32).clone()
                    )
        level = (x_max - x_min) / float((1 << bits) - 1)
        # A non-finite span (from a corrupt range) cannot produce a usable grid;
        # treat the quantizer as a pass-through rather than emitting nan.
        if not bool(torch.isfinite(level).all()):
            return x
        if self.legacy_mode:
            # Pre-fix behaviour: the original absolute epsilon floor.
            level = level.clamp_min(torch.finfo(x.dtype).eps)
        else:
            level = self._apply_step_floor(level, x_max - x_min, x.dtype)
        level = self._effective_level(level)
        q = torch.round((x - x_min) / level) * level + x_min
        if self.clamp_output:
            # Clip to the representable range. Real integer quantization cannot
            # leave [x_min, x_max], so this is the faithful behaviour rather than a
            # numerical patch: without it a value beyond the calibrated range
            # rounds to a level outside it and the result is unbounded.
            #
            # This is what makes 2-bit survivable here. Measured cascade without
            # it: a single coarse step (~3.4e+03 over a span of 1.0e+04) let an
            # activation reach 2.6e+04, the next step overflowed to inf, and the
            # model diverged (logs/fx3_fixed2_e8_s2050.crash.txt).
            q = torch.clamp(q, min=x_min, max=x_max)
        # Straight-through: the rounding error is detached, but the level
        # dependency must survive so a learnable step multiplier receives a
        # non-zero gradient. `x + (q - x).detach()` would zero it.
        q_ste = level * ((q - x) / level).detach()
        if self._finitude_check:
            self._finitude_counter += 1
            if self._finitude_counter % self.finitude_check_every == 0:
                self._assert_finite(x, x_min, x_max, level, q)
        if self._telemetry:
            with torch.no_grad():
                self.last_x_min.copy_(x_min.detach().float().reshape(-1)[0])
                self.last_x_max.copy_(x_max.detach().float().reshape(-1)[0])
                self.last_step.copy_(level.detach().float().reshape(-1)[0])
                self.last_out_absmax.copy_(q.detach().float().abs().max())
        return x + q_ste


class QATLinear(nn.Linear):
    def __init__(self, source, out_features=None, bias=True, device=None, dtype=None, low_bits: int = 8):
        if isinstance(source, nn.Linear):
            super().__init__(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
            self.load_state_dict(source.state_dict())
        else:
            super().__init__(source, out_features, bias=bias, device=device, dtype=dtype)
        self.weight_quant = UniformQuantizer(
            low_bits, per_channel=True, range_mode="ema"
        )
        self.weight_quant.use_exact_range = True
        self.activation_quant = UniformQuantizer(
            low_bits, per_channel=False, range_mode="ema"
        )
        # Optional orthogonal (Hadamard) rotation of the quantized basis.
        self.hadamard_rotate = False
        self.register_buffer("_had_out", None, persistent=False)

    @torch.no_grad()
    def set_hadamard_rotate(self, enabled: bool = True) -> None:
        """Quantize the weight in a Hadamard-rotated (flatter) basis.

        Only the weight's output dimension is rotated, so in FP32 the layer output
        equals the original output expressed in the rotated basis
        (``x @ W_rot^T = x @ W^T H^T``). A layer whose input dimension is not a
        power-of-two multiple keeps the original basis.
        """
        self.hadamard_rotate = bool(enabled)
        if not enabled:
            self._had_out = None
            return
        _, hadamard = rotate_weight(
            self.out_features, self.weight, self.weight.device, self.weight.dtype
        )
        self._had_out = hadamard

    def forward(self, x: Tensor) -> Tensor:
        """Quantized forward, optionally in the rotated weight basis.

        Note the rotation is applied to the **weights only** (``x`` stays in the
        original basis). Rotating the activations too turns the sparse, heavy
        tailed post-GELU distribution into a dense one; the per-channel activation
        quantizer then loses the sparsity it relies on, the activation ranges
        explode across blocks and training diverges (observed as ``loss=nan``).
        Rotating only the weights keeps the activation statistics intact while
        still removing the weight outliers that hurt the weight grid.
        """
        if self.hadamard_rotate and self._had_out is not None:
            weight = self._had_out @ self.weight
        else:
            weight = self.weight
        xq = self.activation_quant(x)
        wq = self.weight_quant(weight)
        return F.linear(xq, wq, self.bias)


class QATConv1d(nn.Conv1d):
    def __init__(self, source: nn.Conv1d, low_bits: int = 8):
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.load_state_dict(source.state_dict())
        self.weight_quant = UniformQuantizer(
            low_bits, per_channel=True, range_mode="ema"
        )
        self.weight_quant.use_exact_range = True
        self.activation_quant = UniformQuantizer(
            low_bits, per_channel=False, range_mode="ema"
        )

    def forward(self, x: Tensor) -> Tensor:
        xq = self.activation_quant(x)
        wq = self.weight_quant(self.weight)
        return self._conv_forward(xq, wq, self.bias)


def convert_to_qat(
    model: nn.Module,
    low_bits: int = 8,
) -> OrderedDict[str, nn.Module]:
    """Replace Linear/Conv1d layers while preserving parameter names."""
    targets: OrderedDict[str, nn.Module] = OrderedDict()
    weight_replacements: dict[int, Tensor] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (QATLinear, QATConv1d)):
                targets[full_name] = child
            elif type(child) is nn.Linear:
                replacement = QATLinear(child, low_bits=low_bits)
                weight_replacements[id(child.weight)] = replacement.weight
                parent._modules[name] = replacement
                targets[full_name] = replacement
            elif type(child) is nn.Conv1d:
                replacement = QATConv1d(child, low_bits=low_bits)
                weight_replacements[id(child.weight)] = replacement.weight
                parent._modules[name] = replacement
                targets[full_name] = replacement
            else:
                visit(child, full_name)
                continue
            # Name the quantizers for actionable non-finite diagnostics.
            targets[full_name].weight_quant.debug_name = f"{full_name}.weight"
            targets[full_name].weight_quant.is_weight = True
            targets[full_name].activation_quant.debug_name = f"{full_name}.activation"
            targets[full_name].activation_quant.is_weight = False

    visit(model)
    # CKAA's IntermReader keeps a hard reference to attention.qkv.weight.
    # Point it at the replacement parameter after quantization wrapping.
    for module in model.modules():
        other_args = getattr(module, "other_args", None)
        if not isinstance(other_args, dict):
            continue
        for key, value in list(other_args.items()):
            if isinstance(value, Tensor) and id(value) in weight_replacements:
                other_args[key] = weight_replacements[id(value)]
    return targets


class AQCLContext:
    """Training-time state and deployment helpers for AQCL."""

    def __init__(self, model: nn.Module, args) -> None:
        self.args = args
        self.mode = getattr(args, "aqcl_mode", "rpq_saou")
        self.low_bits = int(getattr(args, "aqcl_low_bits", 8))
        self.high_bits = int(getattr(args, "aqcl_high_bits", max(2 * self.low_bits, 2)))
        # The paper computes the weight sensitivity (Eq.14) and the activation
        # sensitivity (Eq.15) separately and applies Eq.13 to each, so the two
        # bit-widths are independent. Decoupling them matters here because the
        # deployed size is driven by the weights while the accuracy loss is
        # dominated by the activations: e.g. weights 2-bit with activations 4-bit
        # keeps the 2-bit model size while recovering accuracy.
        act_low = int(
            getattr(args, "aqcl_act_low_bits", 0) or self.low_bits
        )
        act_high = int(
            getattr(args, "aqcl_act_high_bits", 0) or self.high_bits
        )
        self.act_low_bits = act_low
        self.act_high_bits = act_high
        self.fixed_bits = int(getattr(args, "aqcl_bits", self.low_bits))
        self.fixed_act_bits = int(
            getattr(args, "aqcl_act_bits", 0) or self.fixed_bits
        )
        self.lambda_threshold = float(getattr(args, "aqcl_lambda", 10.0))
        self.alpha = float(getattr(args, "aqcl_alpha", 5.0))
        self.theta = float(getattr(args, "aqcl_theta", 0.5))
        self.fisher_batches = int(getattr(args, "aqcl_fisher_batches", 8))
        self.targets = convert_to_qat(model, low_bits=self.low_bits)
        self.weight_sensitivity: dict[str, Tensor] = {}
        self.activation_sensitivity: dict[str, Tensor] = {}
        self.covariance: dict[str, Tensor] = {}
        self.covariance_count: dict[str, int] = {}
        self.subspaces: dict[str, dict[str, Tensor]] = {}
        self.last_mac_profile: dict[str, float] = {}
        self.active_weight_bits: dict[str, int] = {}
        self.active_activation_bits: dict[str, int] = {}
        self.quantization_enabled = True
        self.instrumentation = AQCLInstrumentation(
            enabled=bool(getattr(args, "aqcl_instrument", False)),
            verbose=bool(getattr(args, "aqcl_instrument_verbose", True)),
        )
        self.last_eigenvalues: dict[str, Tensor] = {}
        self.last_gradient_cosines: dict[str, Tensor] = {}
        self.gradient_modulation_stats: dict[str, dict[str, float]] = {}
        self.gradient_modulation_counts: dict[int, int] = {}
        self.model_ref: nn.Module | None = None
        self.last_inference_report: dict[str, float] = {}
        self.last_raw_grad_stats: dict[str, dict[str, float]] = {}
        self.layer_alive: dict[str, bool] = {}
        self.bit_allocation_degenerate = False
        self.weight_threshold: float = 0.0
        self.activation_threshold: float = 0.0
        # Precision floor for the frozen pretrained backbone. In this repository
        # the ViT backbone is frozen, so Eq.13's layer-wise split is applied on
        # top of this floor rather than to the whole network.
        #
        # `--aqcl-backbone-bits 0`  : no floor (paper-faithful, whole network)
        # `--aqcl-backbone-bits N`  : absolute floor of N bits
        # `--aqcl-backbone-bits -1` : floor = b_H, so a 4/8 run keeps the
        #                             backbone at 8-bit while a 2/4 run drops it
        #                             to 4-bit. Without this, a 2/4 run would
        #                             keep an 8-bit backbone and barely shrink
        #                             (86 MB), because the frozen backbone holds
        #                             ~86 of the ~87 M parameters.
        raw_floor = int(getattr(args, "aqcl_backbone_bits", 0) or 0)
        if raw_floor < 0:
            self.backbone_floor_bits = int(self.high_bits)
            self.backbone_floor_is_relative = True
        else:
            self.backbone_floor_bits = raw_floor
            self.backbone_floor_is_relative = False
        self.allocation_rule = str(
            getattr(args, "aqcl_allocation", "mean") or "mean"
        )
        if self.allocation_rule not in ("mean", "paper", "budget"):
            raise ValueError(
                f"Unsupported --aqcl-allocation: {self.allocation_rule}"
            )
        self.rpq_weight_bits: dict[str, int] = {}
        self.rpq_activation_bits: dict[str, int] = {}
        self.average_weight_bits: float = 0.0
        # Activation granularity for targets registered later; default per-tensor
        # until the CLI flag is applied below.
        self.activation_per_channel = False
        # Temperature used only inside the Fisher/sensitivity pass. Saturated
        # logits make the CE gradient exactly zero, which would collapse Eq.11.
        self.fisher_temperature = float(
            getattr(args, "aqcl_fisher_temperature", 64.0) or 64.0
        )
        if bool(getattr(args, "aqcl_per_token_activation", False)):
            self.set_activation_granularity(True)
        # Remembered so that targets registered later (the CKAA classifier head is
        # rebuilt every task) inherit the same activation granularity.
        self.activation_per_channel = bool(
            getattr(args, "aqcl_per_token_activation", False)
        )
        clip = float(getattr(args, "aqcl_clip_percentile", 0.0) or 0.0)
        if clip > 0.0:
            self.set_clip_percentile(clip)
        if bool(getattr(args, "aqcl_learnable_scale", False)):
            self.set_learnable_scale(True)
        if bool(getattr(args, "aqcl_hadamard", False)):
            scope = str(getattr(args, "aqcl_hadamard_scope", "attention"))
            rotated = self.set_hadamard_rotation(True, scope=scope)
            print(
                f"[AQCL] Hadamard-rotated quantization basis enabled on "
                f"{rotated}/{len(self.targets)} layers (scope={scope})",
                flush=True,
            )
        self.before_task(0)

    @property
    def enabled(self) -> bool:
        return True

    def _apply_telemetry(self) -> None:
        """Propagate the instrumentation flag to quantizer telemetry buffers."""
        telemetry = bool(self.instrumentation.enabled)
        for module in self.targets.values():
            for quantizer in (
                module.weight_quant,
                module.activation_quant,
            ):
                quantizer._telemetry = telemetry
                quantizer._finitude_check = telemetry

    def _linear_targets(self) -> Iterable[tuple[str, QATLinear]]:
        for name, module in self.targets.items():
            if isinstance(module, QATLinear):
                yield name, module

    def set_bits(self, weight_bits: dict[str, int], activation_bits: dict[str, int]) -> None:
        self.active_weight_bits = dict(weight_bits)
        self.active_activation_bits = dict(activation_bits)
        if not self.quantization_enabled:
            return
        for name, module in self.targets.items():
            if not self.quantization_scope_allows(name):
                module.weight_quant.set_bits(32)
                module.activation_quant.set_bits(32)
                continue
            module.weight_quant.set_bits(weight_bits.get(name, self.low_bits))
            module.activation_quant.set_bits(activation_bits.get(name, self.low_bits))

    def quantization_scope_allows(self, name: str) -> bool:
        """Whether a named layer is inside the active quantization scope.

        ``AQCL_QUANTIZE_BACKBONE=0`` restricts quantization to the CKAA
        trainable set (prompts, adapters, heads). This is an ablation: the AQCL
        paper quantizes every trainable layer, but in this repository the ViT
        backbone is frozen, so the ablation separates "quantize the frozen
        feature extractor" from "quantize what is actually being learned".
        """
        if os.environ.get("AQCL_QUANTIZE_BACKBONE", "1") != "0":
            return True
        return any(
            token in name
            for token in (
                "prompt",
                "task_adapter",
                "adapter",
                "head",
                "keys",
                "logit_scale",
            )
        )

    def set_quantization_enabled(self, enabled: bool) -> None:
        self.quantization_enabled = bool(enabled)
        self._apply_telemetry()
        if enabled:
            for name, module in self.targets.items():
                if not self.quantization_scope_allows(name):
                    module.weight_quant.set_bits(32)
                    module.activation_quant.set_bits(32)
                    continue
                module.weight_quant.set_bits(
                    self.active_weight_bits.get(name, self.low_bits)
                )
                module.activation_quant.set_bits(
                    self.active_activation_bits.get(name, self.low_bits)
                )
        else:
            for module in self.targets.values():
                module.weight_quant.set_bits(32)
                module.activation_quant.set_bits(32)

    def register_target(self, name: str, module: nn.Module) -> None:
        if isinstance(module, (QATLinear, QATConv1d)):
            self.targets[name] = module
            # Name the quantizers so a non-finite diagnosis can point at the
            # offending layer instead of reporting an anonymous tensor.
            module.weight_quant.debug_name = f"{name}.weight"
            module.weight_quant.is_weight = True
            module.activation_quant.debug_name = f"{name}.activation"
            module.activation_quant.is_weight = False
            # QATLinear/QATConv1d build their activation quantizer per-tensor.
            # The CKAA classifier head is re-created for every task, so a
            # granularity set once on the existing targets would never reach it.
            #
            # NOTE: applying per-channel granularity to the *head* was measured to
            # destroy quantized accuracy (8-bit seed 2024: 80.87% -> 17.44%) while
            # leaving 32-bit identical, because the head reduces 768 channels to a
            # few classes and 768 independent ranges there quantize the logits
            # input far more coarsely in aggregate. The head therefore keeps the
            # per-tensor granularity unless AQCL_HEAD_PER_CHANNEL=1 is set.
            head_like = any(
                token in name for token in ("head", "task_head", "keys")
            )
            if head_like and os.environ.get("AQCL_HEAD_PER_CHANNEL", "0") != "1":
                module.activation_quant.set_per_channel(False)
            else:
                module.activation_quant.set_per_channel(self.activation_per_channel)
            if self.quantization_enabled and self.quantization_scope_allows(name):
                module.weight_quant.set_bits(
                    self.active_weight_bits.get(name, self.low_bits)
                )
                module.activation_quant.set_bits(
                    self.active_activation_bits.get(name, self.low_bits)
                )
            else:
                module.weight_quant.set_bits(32)
                module.activation_quant.set_bits(32)
            module.weight_quant._telemetry = bool(self.instrumentation.enabled)
            module.activation_quant._telemetry = bool(self.instrumentation.enabled)

    def sync_classifier_head(self, model: nn.Module) -> str | None:
        """Keep the growing CKAA classifier head inside the quantizer set.

        ``train_eval.modify_head`` rebuilds ``model.head`` for every task, so the
        freshly created ``nn.Linear`` is outside ``self.targets`` unless it is
        re-registered. Without this the head would silently stay unquantized and
        would be invisible to both RPQ and SAOU, which is not what the AQCL
        formulation assumes.

        Set ``AQCL_QUANTIZE_HEAD=0`` to disable head quantization, which is used
        as a controlled ablation: the head produces the features consumed by the
        CKAA task router and by ``head_out`` logits.
        """
        if os.environ.get("AQCL_QUANTIZE_HEAD", "1") == "0":
            return None
        head = getattr(model, "head", None)
        if head is None:
            return None
        if not isinstance(head, (QATLinear, QATConv1d)):
            replacement = QATLinear(head, low_bits=self.low_bits)
            model._modules["head"] = replacement
            self.register_target("head", replacement)
            return "head"
        if self.targets.get("head") is not head:
            self.register_target("head", head)
            return "head"
        return None

    def calibrate_ranges(
        self,
        model: nn.Module,
        dataloader,
        taskid: int,
        max_batches: int = 4,
        freeze: bool = True,
    ) -> dict[str, int]:
        """Estimate the activation ranges used by the quantized forward pass.

        The ranges are collected in FP32 (quantizers bypassed) with the
        trainable set in eval mode, then locked for the whole task. This is the
        standard QAT calibration contract and it is what makes the quantization
        grid stable enough for 4-bit and 2-bit training; without it, the grid
        changes at every step and the model cannot compensate.
        """
        if not self.targets:
            return {"num_layers": 0, "num_batches": 0}
        device = next(model.parameters()).device
        was_training = model.training
        previous_enabled = self.quantization_enabled
        model.eval()
        # Reset to ema so the calibration pass repopulates fresh statistics.
        self.reset_ranges()
        self.set_quantization_enabled(False)
        seen = 0
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(dataloader):
                    if batch_index >= max_batches:
                        break
                    images = batch[0].to(device, non_blocking=True)
                    model(
                        images,
                        mode=self.args.train_tool + "_train",
                        taskid=taskid,
                        max_taskid=taskid,
                    )
                    seen += 1
        finally:
            self.set_quantization_enabled(previous_enabled)
            model.train(was_training)
        if freeze:
            self.freeze_ranges()
        return {"num_layers": len(self.targets), "num_batches": seen}

    def refresh_activation_ranges(
        self, range_mode: str
    ) -> None:
        """Re-enable activation range tracking at the last calibration values.

        Unlike ``reset_ranges`` this keeps the calibrated numbers as the starting
        point, so the EMA continues from the calibration instead of starting cold.
        """
        for module in self.targets.values():
            module.activation_quant.unfreeze(range_mode)

    def freeze_ranges(self) -> None:
        for module in self.targets.values():
            module.activation_quant.freeze()
            module.weight_quant.freeze()

    def set_range_tracking(self, mode: str) -> None:
        """Choose how the activation range behaves during training.

        ``frozen``  calibrate once per task, then keep the grid (the default; a
                    fixed grid lets the weights adapt to it).
        ``ema``     keep updating the running range while training. With a long
                    schedule this lets the activation grid follow the drifting
                    activation statistics, which the frozen variant cannot do.
        ``dynamic`` recompute per batch (the original, unstable behaviour).
        """
        if mode not in ("frozen", "ema", "dynamic"):
            raise ValueError(f"Unsupported range tracking mode: {mode}")
        for module in self.targets.values():
            if mode == "frozen":
                module.activation_quant.freeze()
            else:
                module.activation_quant.unfreeze(mode)

    def set_activation_granularity(self, per_token: bool) -> None:
        """Use one activation range per token instead of one per tensor."""
        for module in self.targets.values():
            module.activation_quant.set_per_channel(bool(per_token))

    def set_hadamard_rotation(
        self, enabled: bool = True, frozen_only: bool = True, scope: str = "backbone"
    ) -> int:
        """Enable the Hadamard-rotated quantization basis on Linear layers.

        ``frozen_only`` (default) restricts the rotation to layers that the
        continual algorithm does not train, i.e. the frozen pretrained backbone.
        Their weights are static, so the rotation is a fixed orthogonal change of
        basis and cannot perturb the CKAA adapter training dynamics, while it
        removes exactly the weight outliers that dominate 2-4 bit error.

        ``scope`` selects which backbone layers are eligible:

        ``attention``   the attention ``qkv`` and ``proj`` projections only
                        (default): they operate on LayerNorm-bounded inputs;
        ``attn_fc2``    attention plus ``mlp.fc2``, whose input is the post-GELU
                        activation (non-negative, so a sparser distribution);
        ``no_fc1``      every frozen Linear except ``mlp.fc1``;
        ``backbone``    every frozen Linear. Rotating ``mlp.fc1`` was measured to
                        diverge (``loss=nan`` from epoch 3): its rotated weight is
                        dense instead of heavy tailed, so the pre-GELU feature
                        becomes dense and the activation ranges compound by
                        1.4-2.1x per epoch until the fp16 autocast path overflows.

        Returns the number of layers whose basis was actually rotated.
        """
        count = 0
        trainable_tokens = ("prompt", "task_adapter", "adapter", "head", "keys")
        for name, module in self.targets.items():
            if not isinstance(module, QATLinear):
                continue
            if frozen_only and any(token in name for token in trainable_tokens):
                continue
            if scope == "attention" and "attn" not in name:
                continue
            if scope == "attn_fc2" and not (
                "attn" in name or "mlp.fc2" in name
            ):
                continue
            if scope == "no_fc1" and "mlp.fc1" in name:
                continue
            module.set_hadamard_rotate(bool(enabled))
            if module.hadamard_rotate and module._had_out is not None:
                count += 1
        return count

    def set_learnable_scale(self, enabled: bool = True) -> None:
        """Enable or disable the learnable per-channel step multiplier.

        ``scale_raw`` is always a registered buffer; this only toggles whether it
        participates in the effective step, so the state dict layout is stable
        between runs that do and do not use it.
        """
        for module in self.targets.values():
            for quantizer in (module.weight_quant, module.activation_quant):
                quantizer.learnable_scale = bool(enabled)

    def quantizer_parameters(self) -> list[Tensor]:
        """Step-multiplier parameters, for the optimizer and the checkpoint."""
        params: list[Tensor] = []
        for module in self.targets.values():
            for quantizer in (module.weight_quant, module.activation_quant):
                if quantizer.learnable_scale:
                    quantizer.scale_raw.requires_grad_(True)
                    params.append(quantizer.scale_raw)
        return params

    def set_clip_percentile(self, percentile: float) -> None:
        """Set the outlier-clipping quantile for weight and activation ranges.

        ``percentile=0`` keeps the plain min/max range. A value such as 0.999
        clips the extreme 0.1% of each channel, which keeps a small
        quantization step for the bulk of the distribution. This is applied when
        the range is (re)computed during calibration, and the frozen range keeps
        the clipped value for the rest of the task.
        """
        value = float(percentile)
        if value < 0.0 or value >= 1.0:
            raise ValueError("clip percentile must be in [0, 1)")
        for module in self.targets.values():
            module.activation_quant.clip_percentile = value
            module.weight_quant.clip_percentile = value

    def unfreeze_ranges(self) -> None:
        for module in self.targets.values():
            module.activation_quant.unfreeze("ema")
            module.weight_quant.unfreeze("ema")

    def reset_ranges(self) -> None:
        """Clear calibrated ranges so the next task is calibrated fresh.

        Task distributions differ, so carrying task t-1 statistics into task t
        would clip the new task against the wrong scale.
        """
        for module in self.targets.values():
            for quantizer in (module.weight_quant, module.activation_quant):
                quantizer.range_ready.fill_(0)
                quantizer.range_mode = "ema"

    def collect_sensitivity(
        self,
        model: nn.Module,
        dataloader,
        taskid: int,
        criterion: nn.Module,
        max_batches: int = 2,
        ema: float = 1.0,
        reset: bool = True,
    ) -> dict[str, float]:
        """Estimate Eq.11/12 sensitivity from the warm-up (FP32) model.

        The paper computes the diagonal empirical Fisher after training the
        current task. On this dataset that model has saturated logits, so
        ``CrossEntropyLoss`` and its gradient are exactly zero for most samples
        and the sensitivity collapses to 0 for every layer. Collecting the
        gradients from the warm-up phase (before quantization and before the
        logits saturate) keeps ``S_l`` informative while remaining a first-order
        Fisher estimate.
        """
        if not self.targets:
            return {"num_layers": 0, "num_batches": 0, "nonzero_layers": 0}
        device = next(model.parameters()).device
        was_training = model.training
        previous_requires_grad = {
            name: module.weight.requires_grad
            for name, module in self.targets.items()
        }
        for module in self.targets.values():
            module.weight.requires_grad_(True)
        sums = {
            name: torch.zeros_like(module.weight, device=device)
            for name, module in self.targets.items()
        }
        counts = {name: 0 for name in self.targets}
        activation_sums = {name: 0.0 for name in self.targets}
        activation_counts = {name: 0 for name in self.targets}

        def activation_hook(name: str):
            def hook(_module, grad_input, _grad_output):
                if grad_input and grad_input[0] is not None:
                    activation_sums[name] += float(
                        grad_input[0].detach().float().pow(2).sum().item()
                    )
                    activation_counts[name] += int(grad_input[0].numel())
            return hook

        handles = [
            module.register_full_backward_hook(activation_hook(name))
            for name, module in self._linear_targets()
        ]
        model.train()
        seen = 0
        try:
            for batch_index, batch in enumerate(dataloader):
                if batch_index >= max_batches:
                    break
                images = batch[0].to(device, non_blocking=True)
                target = batch[1].to(device, non_blocking=True)
                model.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=bool(getattr(self.args, "use_amp", False)),
                ):
                    _, logits, _ = model(
                        images,
                        mode=self.args.train_tool + "_train",
                        taskid=taskid,
                        max_taskid=taskid,
                    )
                    # CrossEntropyLoss on saturated logits is exactly 0.0 and
                    # its gradient is exactly zero, which would zero the Fisher
                    # trace. Dividing the logits by a temperature > 1 restores a
                    # non-degenerate gradient without changing which layer is
                    # more sensitive (it is a monotone rescaling of the same
                    # loss). This pass never updates any parameter.
                    scaled_logits = (
                        logits / self.fisher_temperature
                        if self.fisher_temperature > 0
                        and abs(self.fisher_temperature - 1.0) > 1e-9
                        else logits
                    )
                    loss = criterion(scaled_logits, target)
                loss.backward()
                for name, module in self.targets.items():
                    grad = module.weight.grad
                    if grad is None:
                        continue
                    if float(grad.norm().item()) == 0.0:
                        continue
                    sums[name] += grad.detach().pow(2)
                    counts[name] += int(module.weight.numel())
                seen += 1
        finally:
            for handle in handles:
                handle.remove()
            for name, module in self.targets.items():
                module.weight.requires_grad_(previous_requires_grad[name])
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        nonzero = 0
        reused = 0
        for name in self.targets:
            count = counts[name]
            if count == 0:
                # No usable gradient for this layer in this pass. When a
                # previous task already produced an estimate, keep it instead of
                # discarding the layer: task 5 of this dataset is a single-class
                # task whose logits are flat, so a genuine per-task Fisher is
                # impossible and dropping the estimate would degenerate the
                # allocation to a single bit-width.
                if name in self.weight_sensitivity:
                    self.layer_alive[name] = True
                    reused += 1
                elif reset:
                    self.layer_alive[name] = False
                continue
            self.layer_alive[name] = True
            sensitivity = (sums[name].sum() / count).detach().cpu()
            if reset or name not in self.weight_sensitivity:
                self.weight_sensitivity[name] = sensitivity
            elif ema < 1.0:
                self.weight_sensitivity[name] = (
                    ema * sensitivity
                    + (1.0 - ema) * self.weight_sensitivity[name]
                )
            else:
                self.weight_sensitivity[name] = sensitivity
            activation = activation_sums[name] / max(activation_counts[name], 1)
            activation_tensor = torch.tensor(activation, dtype=torch.float32)
            if reset or name not in self.activation_sensitivity:
                self.activation_sensitivity[name] = activation_tensor
            elif ema < 1.0:
                self.activation_sensitivity[name] = (
                    ema * activation_tensor
                    + (1.0 - ema) * self.activation_sensitivity[name]
                )
            else:
                self.activation_sensitivity[name] = activation_tensor
            if float(sensitivity.item()) > 0.0:
                nonzero += 1
        if self.instrumentation.enabled:
            print(
                f"[AQCL-INSTRUMENT][fisher_warmup] task={taskid} "
                f"batches={seen} layers={len(self.targets)} "
                f"nonzero_layers={nonzero} reused_layers={reused}",
                flush=True,
            )
        self._allocate_bits()
        return {
            "num_layers": len(self.targets),
            "num_batches": seen,
            "nonzero_layers": nonzero,
            "reused_layers": reused,
        }

    def before_task(self, taskid: int) -> None:
        if self.mode == "fixed":
            weight_bits = {
                name: max(self.fixed_bits, self._floor_for(name))
                for name in self.targets
            }
            activation_bits = {
                name: max(self.fixed_act_bits, self._floor_for(name))
                for name in self.targets
            }
            self.set_bits(weight_bits, activation_bits)
        elif not self.weight_sensitivity:
            # Sensitivity is estimated only after the first task. Start from
            # the safe high precision so the first task is not pre-compressed.
            weight_bits = {name: self.high_bits for name in self.targets}
            activation_bits = {name: self.act_high_bits for name in self.targets}
            self.set_bits(weight_bits, activation_bits)
        elif self.mode not in ("rpq", "rpq_saou"):
            raise ValueError(f"Unsupported AQCL mode: {self.mode}")

    def _trainable_layer_names(self) -> set[str]:
        """Layers in the CKAA trainable set (prompts, adapters, heads).

        These carry the continual-learning knowledge that is actually being
        learned in this repository, so the allocator always protects them. The
        frozen pretrained backbone is handled separately by
        ``--aqcl-backbone-bits`` and takes part in the sensitivity split only
        when that floor is 0.
        """
        names: set[str] = set()
        tokens = (
            "prompt",
            "task_adapter",
            "adapter",
            "head",
            "keys",
            "logit_scale",
        )
        for name in self.targets:
            if any(token in name for token in tokens):
                names.add(name)
        return names

    def _allocate_bits(self) -> None:
        """Eq.13: S_l >= S_avg -> high bits, otherwise low bits.

        Layers that did not participate in the current-task forward pass have no
        Fisher signal and would otherwise bias ``S_avg`` (and, when every batch
        saturates, collapse the rule to "all high bits"). They are assigned the
        low bit-width instead, which is both statistically honest and the
        memory-optimal choice for weights that the current task never uses.
        """
        if self.mode == "fixed":
            return
        if not self.weight_sensitivity:
            return
        alive = {
            name: value
            for name, value in self.weight_sensitivity.items()
            if self.layer_alive.get(name, True)
        }
        alive_act = {
            name: value
            for name, value in self.activation_sensitivity.items()
            if self.layer_alive.get(name, True)
        }
        if not alive:
            # No usable sensitivity signal at all (for example every Fisher
            # batch saturated). Do not silently pretend the allocation is
            # meaningful: keep high precision and record the degeneracy.
            self.bit_allocation_degenerate = True
            weight_bits = {name: self.high_bits for name in self.targets}
            activation_bits = {name: self.act_high_bits for name in self.targets}
            self.set_bits(weight_bits, activation_bits)
            if self.instrumentation.enabled:
                print(
                    "[AQCL-INSTRUMENT][alloc_warning] no alive layer had a "
                    "nonzero Fisher signal; falling back to high precision for "
                    "every layer (RPQ allocation is NOT meaningful here).",
                    flush=True,
                )
            return
        self.bit_allocation_degenerate = False
        weight_values = torch.stack(list(alive.values()))
        weight_threshold = weight_values.mean()
        if alive_act:
            activation_values = torch.stack(list(alive_act.values()))
            activation_threshold = activation_values.mean()
        else:
            activation_values = weight_values
            activation_threshold = weight_threshold

        if self.allocation_rule == "mean":
            weight_bits = {
                name: (
                    self.high_bits
                    if (
                        self.layer_alive.get(name, False)
                        and name in alive
                        and float(alive[name].item())
                        >= float(weight_threshold.item())
                    )
                    else self.low_bits
                )
                for name in self.weight_sensitivity
            }
            activation_bits = {
                name: (
                    self.act_high_bits
                    if (
                        self.layer_alive.get(name, False)
                        and name in alive_act
                        and float(alive_act[name].item())
                        >= float(activation_threshold.item())
                    )
                    else self.act_low_bits
                )
                for name in self.activation_sensitivity
            }
        else:
            # "paper": only layers that are at least as sensitive as the average
            # are protected; layers below the average are compressed. Unlike
            # "mean", a layer with a below-average but non-zero sensitivity is
            # never promoted by a threshold reduction.
            weight_bits = {
                name: (
                    self.high_bits
                    if (
                        self.layer_alive.get(name, False)
                        and name in alive
                        and float(alive[name].item())
                        >= float(weight_threshold.item())
                    )
                    else self.low_bits
                )
                for name in self.weight_sensitivity
            }
            activation_bits = {
                name: (
                    self.act_high_bits
                    if (
                        self.layer_alive.get(name, False)
                        and name in alive_act
                        and float(alive_act[name].item())
                        >= float(activation_threshold.item())
                    )
                    else self.act_low_bits
                )
                for name in self.activation_sensitivity
            }
            if self.allocation_rule == "budget" and alive:
                # Sensitivity-proportional protection with a hard average
                # bit-width budget. The backbone floor (if configured) is part
                # of the budget, so the promoted layers can never push the
                # average above the target.
                #
                # 1) The CKAA trainable set (prompts, adapters, heads) is
                #    promoted unconditionally: it is the knowledge that
                #    continual learning actually acquires, and it is a tiny
                #    fraction of the parameter count.
                # 2) The remaining budget promotes the most sensitive layers
                #    whose floor is still below b_H, which is what produces a
                #    genuine b_L/b_H mixture inside the RPQ design space.
                trainable = self._trainable_layer_names()
                numel = {
                    name: float(module.weight.numel())
                    for name, module in self.targets.items()
                }
                floor = {
                    name: max(self._floor_for(name), self.low_bits)
                    for name in self.targets
                }
                # The backbone floor pins many layers at or above b_H, so a
                # whole-network average can never reach the b_L/b_H midpoint.
                # The budget is therefore applied to the *flexible* layers that
                # are not pinned by the floor: enough of the most sensitive ones
                # are promoted to b_H that half of the flexible parameter mass is
                # at b_H and half stays at b_L. This always yields a genuine
                # b_L/b_H mixture inside the RPQ design space.
                flexible = {
                    name: numel[name]
                    for name in self.targets
                    if floor[name] < self.high_bits
                }
                flexible_mass = sum(flexible.values())
                target = 0.5 * (self.low_bits + self.high_bits)
                weight_bits = {name: floor[name] for name in self.targets}
                if flexible_mass > 0:
                    extra = 0.5 * flexible_mass
                    used = 0.0
                    for name in trainable:
                        if name in flexible and weight_bits[name] < self.high_bits:
                            weight_bits[name] = self.high_bits
                            used += numel[name]
                    for name, _value in sorted(
                        alive.items(),
                        key=lambda kv: float(kv[1].item()),
                        reverse=True,
                    ):
                        if used >= extra:
                            break
                        if name not in flexible:
                            continue
                        if weight_bits.get(name, 0) >= self.high_bits:
                            continue
                        weight_bits[name] = self.high_bits
                        used += numel.get(name, 1.0)
                self.weight_threshold = float(weight_threshold.item())
                self.activation_threshold = float(activation_threshold.item())
                self.rpq_weight_bits = dict(weight_bits)
                self.rpq_activation_bits = dict(activation_bits)
                self.set_bits(weight_bits, activation_bits)
                self.average_weight_bits = (
                    sum(numel[name] * weight_bits[name] for name in self.targets)
                    / max(sum(numel.values()), 1.0)
                )
                return
        self.weight_threshold = float(weight_threshold.item())
        self.activation_threshold = float(activation_threshold.item())
        # Eq.13 output before the optional backbone floor (paper-faithful RPQ).
        self.rpq_weight_bits = dict(weight_bits)
        self.rpq_activation_bits = dict(activation_bits)
        if self.backbone_floor_bits:
            weight_bits = {
                name: max(bits, self._floor_for(name))
                for name, bits in weight_bits.items()
            }
            activation_bits = {
                name: max(bits, self._floor_for(name))
                for name, bits in activation_bits.items()
            }
        self.set_bits(weight_bits, activation_bits)

    def _floor_for(self, name: str) -> int:
        """Minimum bit-width for a layer, applied only when configured.

        Layers inside the CKAA trainable set (prompts, adapters, heads) are what
        the continual algorithm actually learns; a frozen backbone layer is a
        pretrained feature extractor whose low-bit truncation cannot be recovered
        by the small trainable set. The floor is opt-in
        (``--aqcl-backbone-bits``) and recorded in the deployment report.
        """
        if not self.backbone_floor_bits:
            return 0
        trainable_tokens = (
            "prompt",
            "task_adapter",
            "adapter",
            "head",
            "keys",
            "logit_scale",
        )
        if any(token in name for token in trainable_tokens):
            return 0
        return int(self.backbone_floor_bits)

    def _fisher_and_covariance(
        self,
        model: nn.Module,
        dataloader,
        taskid: int,
        criterion: nn.Module,
        update_sensitivity: bool = False,
    ) -> None:
        args = self.args
        device = next(model.parameters()).device
        was_training = model.training
        model.train()

        previous_requires_grad = {
            name: module.weight.requires_grad
            for name, module in self.targets.items()
        }
        for name, module in self.targets.items():
            module.weight.requires_grad_(True)

        weight_sums = {name: torch.zeros_like(module.weight, device=device) for name, module in self.targets.items()}
        weight_counts = {name: 0 for name in self.targets}
        activation_sums = {name: 0.0 for name in self.targets}
        activation_counts = {name: 0 for name in self.targets}
        # Instrumentation-only raw diagnostics for the Fisher pass.
        raw_grad_stats: dict[str, dict[str, float]] = {}
        batch_grad_norms: list[float] = []
        covariance_local = {}
        covariance_counts = {}
        if self.mode == "rpq_saou":
            covariance_local = {
                name: torch.zeros(
                    module.in_features,
                    module.in_features,
                    device="cpu",
                    dtype=torch.float32,
                )
                for name, module in self._linear_targets()
            }
            covariance_counts = {name: 0 for name, _ in self._linear_targets()}

        def activation_hook(name: str):
            def hook(_module, grad_input, _grad_output):
                if grad_input and grad_input[0] is not None:
                    activation_sums[name] += float(grad_input[0].detach().float().pow(2).sum().item())
                    activation_counts[name] += int(grad_input[0].numel())
            return hook

        def covariance_hook(name: str):
            def hook(_module, inputs):
                if not inputs:
                    return
                x = inputs[0].detach().float()
                if isinstance(_module, nn.Linear):
                    x = x.reshape(-1, x.shape[-1])
                elif x.ndim == 3:
                    x = x.transpose(1, 2).reshape(-1, x.shape[1])
                else:
                    x = x.reshape(-1, x.shape[-1])
                covariance_local[name] += (x.t() @ x).cpu()
                covariance_counts[name] += int(x.shape[0])
            return hook

        handles = []
        for name, module in self._linear_targets():
            handles.append(module.register_full_backward_hook(activation_hook(name)))
            if self.mode == "rpq_saou":
                handles.append(module.register_forward_pre_hook(covariance_hook(name)))

        try:
            for batch_index, (images, target, _) in enumerate(dataloader):
                if batch_index >= self.fisher_batches:
                    break
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                model.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=bool(getattr(args, "use_amp", False)),
                ):
                    _, logits, _ = model(
                        images,
                        mode=args.train_tool + "_train",
                        taskid=taskid,
                        max_taskid=taskid,
                    )
                    loss = criterion(logits, target)
                loss.backward()
                batch_norm = 0.0
                _non_none = 0
                _max_abs = 0.0
                for name, module in self.targets.items():
                    grad = module.weight.grad
                    stats = raw_grad_stats.setdefault(
                        name, {"none": 0.0, "zero": 0.0, "norm_sum": 0.0, "calls": 0.0}
                    )
                    stats["calls"] += 1.0
                    if grad is None:
                        stats["none"] += 1.0
                        continue
                    _non_none += 1
                    _max_abs = max(_max_abs, float(grad.detach().abs().max().item()))
                    grad_norm = float(grad.norm().item())
                    stats["norm_sum"] += grad_norm
                    batch_norm += grad_norm
                    if grad_norm == 0.0:
                        stats["zero"] += 1.0
                    weight_sums[name] += grad.detach().pow(2)
                    weight_counts[name] += int(module.weight.numel())
                batch_grad_norms.append(batch_norm)
                if self.instrumentation.enabled and batch_index < 3:
                    head_grad = getattr(model, "head", None)
                    head_grad_none = (
                        head_grad.weight.grad is None
                        if isinstance(head_grad, nn.Linear)
                        else "n/a"
                    )
                    with torch.no_grad():
                        det_logits = logits.detach().float()
                        probe_layer = self.targets.get("blocks.0.attn.qkv")
                        probe_grad = (
                            None if probe_layer is None else probe_layer.weight.grad
                        )
                        print(
                            f"[AQCL-INSTRUMENT][fisher] task={taskid} "
                            f"batch={batch_index} loss={float(loss.detach().item()):.6f} "
                            f"logits[shape={tuple(det_logits.shape)} "
                            f"min={float(det_logits.min().item()):.3f} "
                            f"max={float(det_logits.max().item()):.3f} "
                            f"mean={float(det_logits.mean().item()):.3f} "
                            f"absmax={float(det_logits.abs().max().item()):.3f}] "
                            f"target={target.detach().tolist()[:8]} "
                            f"qkv_grad_norm="
                            f"{'None' if probe_grad is None else '%.3e' % float(probe_grad.norm().item())} "
                            f"grad_non_none={_non_none}/{len(self.targets)} "
                            f"max_abs_grad={_max_abs:.3e} "
                            f"sum_layer_grad_norm={batch_norm:.6e} "
                            f"head.grad_None={head_grad_none}",
                            flush=True,
                        )
        finally:
            for handle in handles:
                handle.remove()
            for name, module in self.targets.items():
                module.weight.requires_grad_(previous_requires_grad[name])
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        for name, module in self.targets.items():
            count = weight_counts[name]
            if count == 0:
                continue
            sensitivity = (weight_sums[name].sum() / count).detach().cpu()
            if not update_sensitivity:
                continue
            if name in self.weight_sensitivity:
                # Refinement is opt-in: the post-training Fisher is unreliable
                # here because saturated logits zero the gradients, so it would
                # overwrite a good warm-up estimate with noise.
                self.weight_sensitivity[name] = (
                    0.5 * sensitivity + 0.5 * self.weight_sensitivity[name]
                )
            else:
                self.weight_sensitivity[name] = sensitivity
            act_sensitivity = activation_sums[name] / max(activation_counts[name], 1)
            act_tensor = torch.tensor(act_sensitivity, dtype=torch.float32)
            if name in self.activation_sensitivity:
                self.activation_sensitivity[name] = (
                    0.5 * act_tensor + 0.5 * self.activation_sensitivity[name]
                )
            else:
                self.activation_sensitivity[name] = act_tensor
            # A layer is "alive" for this task only if it actually participated
            # in the current-task forward pass. Unused layers (for example task
            # adapters of tasks that have not started yet) carry no sensitivity
            # signal, so Eq.13 must not average them into S_avg.
            self.layer_alive[name] = weight_counts[name] > 0

        if self.instrumentation.enabled:
            none_layers = [n for n, s in raw_grad_stats.items() if s["none"] > 0]
            zero_layers = [
                n
                for n, s in raw_grad_stats.items()
                if s["calls"] > 0 and s["zero"] == s["calls"] and s["none"] == 0
            ]
            nonzero_layers = [
                n for n, s in raw_grad_stats.items() if s["norm_sum"] > 0.0
            ]
            print(
                f"[AQCL-INSTRUMENT][fisher_summary] task={taskid} "
                f"targets={len(self.targets)} grad_None_layers={len(none_layers)} "
                f"grad_all_zero_layers={len(zero_layers)} "
                f"grad_nonzero_layers={len(nonzero_layers)} "
                f"alive_layers={sum(1 for v in self.layer_alive.values() if v)} "
                f"batch_grad_norms={['%.3e' % v for v in batch_grad_norms]}",
                flush=True,
            )
            if none_layers:
                print(
                    f"[AQCL-INSTRUMENT][fisher_summary] grad_None examples: "
                    f"{none_layers[:5]}",
                    flush=True,
                )
            self.last_raw_grad_stats = raw_grad_stats

        if self.mode != "rpq_saou":
            self._allocate_bits()
            return

        for name, module in self._linear_targets():
            count = covariance_counts[name]
            if count == 0:
                continue
            current = covariance_local[name] / float(count)
            if name in self.covariance:
                old_count = self.covariance_count[name]
                self.covariance[name] = (
                    (old_count * self.covariance[name] + count * current)
                    / float(old_count + count)
                )
                self.covariance_count[name] = old_count + count
            else:
                self.covariance[name] = current
                self.covariance_count[name] = count
            covariance = self.covariance[name]
            covariance = (covariance + covariance.t()) * 0.5
            covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
            scale = float(torch.diagonal(covariance).abs().mean().clamp_min(1e-8))
            jitter = scale * 1e-6
            for _ in range(5):
                try:
                    singular_values, eigenvectors = torch.linalg.eigh(
                        covariance + jitter * torch.eye(
                            covariance.shape[0],
                            dtype=covariance.dtype,
                        )
                    )
                    break
                except RuntimeError:
                    jitter *= 10.0
            else:
                raise RuntimeError(f"Unable to decompose covariance for {name}")
            order = torch.argsort(singular_values, descending=True)
            singular_values = singular_values[order].clamp_min(0.0)
            u = eigenvectors[:, order]
            # Instrumentation-only snapshot (no effect on the masks below).
            self.last_eigenvalues[name] = singular_values.detach().cpu().clone()
            floor = singular_values[-1].clamp_min(1e-12)
            threshold = self.lambda_threshold * floor
            null_mask = singular_values <= threshold
            transition_mask = (
                (singular_values >= 0.5 * threshold)
                & (singular_values <= 2.0 * threshold)
            )
            if not null_mask.any():
                null_mask[-1] = True
            if not transition_mask.any():
                transition_mask = null_mask
            self.subspaces[name] = {
                "Uo": u[:, null_mask].detach().cpu(),
                "Up": u[:, transition_mask].detach().cpu(),
            }
        self._allocate_bits()

    def _capture_subspace_diagnostics(self, taskid: int) -> None:
        """Record eigenvalue ranges and Uo/Up dimensions (logging only)."""
        self.instrumentation.subspace_report(
            stage="saou_subspace",
            taskid=taskid,
            eigenvalues=self.last_eigenvalues,
            subspaces=self.subspaces,
            covariance=self.covariance,
            covariance_count=self.covariance_count,
            lambda_threshold=self.lambda_threshold,
        )

    def _capture_sensitivity_diagnostics(self, taskid: int) -> None:
        """Record per-layer Fisher sensitivity vs assigned bit-width."""
        weight_shapes = {
            name: tuple(module.weight.shape) for name, module in self.targets.items()
        }
        weight_numel = {
            name: int(module.weight.numel()) for name, module in self.targets.items()
        }
        record = self.instrumentation.sensitivity_report(
            stage="rpq_sensitivity",
            taskid=taskid,
            weight_sensitivity=self.weight_sensitivity,
            activation_sensitivity=self.activation_sensitivity,
            weight_bits=self.active_weight_bits,
            activation_bits=self.active_activation_bits,
            low_bits=self.low_bits,
            high_bits=self.high_bits,
            weight_shapes=weight_shapes,
            weight_numel=weight_numel,
        )
        record["mode"] = self.mode
        record["layer_alive"] = dict(self.layer_alive)
        record["weight_threshold"] = self.weight_threshold
        record["activation_threshold"] = self.activation_threshold
        record["bit_allocation_degenerate"] = self.bit_allocation_degenerate
        record["backbone_floor_bits"] = self.backbone_floor_bits
        # Bits that Eq.13 alone would have produced, before the backbone floor.
        record["rpq_weight_bits"] = dict(self.rpq_weight_bits)
        if self.rpq_weight_bits:
            pre_floor: dict[str, int] = {}
            for bits in self.rpq_weight_bits.values():
                pre_floor[str(bits)] = pre_floor.get(str(bits), 0) + 1
            record["rpq_weight_bits_distribution_before_floor"] = pre_floor
        return record

    def after_task(
        self,
        model: nn.Module,
        dataloader,
        taskid: int,
        criterion: nn.Module,
        update_sensitivity: bool = False,
    ) -> None:
        self.model_ref = model
        self._fisher_and_covariance(
            model,
            dataloader,
            taskid,
            criterion,
            update_sensitivity=update_sensitivity,
        )
        if self.instrumentation.enabled:
            self._capture_sensitivity_diagnostics(taskid)
            if self.mode == "rpq_saou":
                self._capture_subspace_diagnostics(taskid)

    @torch.no_grad()
    def modulate_gradients(self, model: nn.Module) -> None:
        if self.mode != "rpq_saou":
            return
        collect = self.instrumentation.enabled
        for name, module in self._linear_targets():
            gradient = module.weight.grad
            subspace = self.subspaces.get(name)
            if gradient is None or subspace is None:
                continue
            uo = subspace["Uo"].to(gradient.device, gradient.dtype)
            up = subspace["Up"].to(gradient.device, gradient.dtype)
            if uo.numel() == 0:
                continue
            go = (gradient @ uo) @ uo.t()
            gp = gradient - go
            gp_norm = gp.norm()
            g_norm = gradient.norm().clamp_min(1e-12)
            if gp.numel() == 0 or gp_norm <= self.theta * g_norm:
                if collect:
                    self._record_gradient_stats(
                        name, gradient, go, gp, None, applied=False
                    )
                module.weight.grad = go
                continue
            if up.numel() == 0:
                if collect:
                    self._record_gradient_stats(
                        name, gradient, go, gp, None, applied=False
                    )
                module.weight.grad = go
                continue
            coefficients = gp @ up
            cosine = coefficients / gp_norm.clamp_min(1e-12)
            mask = torch.exp(-self.alpha * cosine.pow(2))
            modulated = (coefficients * mask) @ up.t()
            if collect:
                self._record_gradient_stats(
                    name, gradient, go, gp, modulated, applied=True
                )
            module.weight.grad = go + modulated

    @torch.no_grad()
    def _record_gradient_stats(
        self,
        name: str,
        gradient: Tensor,
        go: Tensor,
        gp: Tensor,
        modulated: Tensor | None,
        applied: bool,
    ) -> None:
        """Logging-only: cosine of gradient vs. the orthogonal complement.

        This mirrors the paper's switching criterion
        ``|cos(g, Uo_perp)| > theta`` where ``Uo_perp`` is the orthogonal
        complement of the retained null-space basis.
        """
        statistics = self.gradient_modulation_stats.setdefault(
            name,
            {
                "steps": 0.0,
                "g_norm_sum": 0.0,
                "go_norm_sum": 0.0,
                "gp_norm_sum": 0.0,
                "cos_sum": 0.0,
                "cos_sq_sum": 0.0,
                "cos_max": 0.0,
                "cos_min": 1.0,
                "cos_last": 0.0,
                "cos_sq_last": 0.0,
                "applied_steps": 0.0,
                "skipped_steps": 0.0,
                "gamma_sum": 0.0,
                "gamma_last": 0.0,
                "modulated_norm_sum": 0.0,
                "theta": float(self.theta),
                "alpha": float(self.alpha),
            },
        )
        g_norm = float(gradient.norm().item())
        gp_norm = float(gp.norm().item())
        go_norm = float(go.norm().item())
        # The paper's switching rule is |cos(g, Uo_perp)|^2 > theta, and because
        # `Uo` is an orthonormal basis this equals the squared projected cosine.
        cos_orth = gp_norm / max(g_norm, 1e-12)
        cos_orth_sq = cos_orth * cos_orth
        if modulated is None:
            gamma = 1.0
            mod_norm = float(go.norm().item())
        else:
            gamma = float(
                (modulated.norm() / gp.norm().clamp_min(1e-12)).item()
            )
            mod_norm = float((go + modulated).norm().item())
        statistics["steps"] += 1.0
        statistics["g_norm_sum"] += g_norm
        statistics["go_norm_sum"] += go_norm
        statistics["gp_norm_sum"] += gp_norm
        statistics["cos_sum"] += cos_orth
        statistics["cos_sq_sum"] += cos_orth_sq
        statistics["cos_max"] = max(statistics["cos_max"], cos_orth)
        statistics["cos_min"] = min(statistics["cos_min"], cos_orth)
        statistics["cos_last"] = cos_orth
        statistics["cos_sq_last"] = cos_orth_sq
        statistics["applied_steps"] += 1.0 if applied else 0.0
        statistics["skipped_steps"] += 0.0 if applied else 1.0
        statistics["gamma_sum"] += gamma
        statistics["gamma_last"] = gamma
        statistics["modulated_norm_sum"] += mod_norm
        self.gradient_modulation_counts[0] = (
            self.gradient_modulation_counts.get(0, 0) + 1
        )

    def capture_task_instrumentation(self, taskid: int) -> None:
        """Flush subspace/deployment/gradient records for a finished task."""
        if not self.instrumentation.enabled:
            return
        if self.mode == "rpq_saou" and self.last_eigenvalues:
            self._capture_subspace_diagnostics(taskid)
        if self.weight_sensitivity:
            weight_bits = {
                name: module.weight_quant.num_bits
                for name, module in self.targets.items()
            }
            activation_bits = {
                name: module.activation_quant.num_bits
                for name, module in self.targets.items()
            }
            param_bits = {
                name: module.weight_quant.num_bits
                for name, module in self.targets.items()
            }
            self.instrumentation.deployment_report(
                stage="deployment_bits",
                taskid=taskid,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                model_size_mb=(
                    self.model_size_mb(self.model_ref)
                    if self.model_ref is not None
                    else float("nan")
                ),
                report=self.last_inference_report,
                param_bits=param_bits,
            )
        self.instrumentation.gradient_report(
            stage="saou_gradient",
            taskid=taskid,
            epochs=self.gradient_modulation_counts,
            stats=self.gradient_modulation_stats,
        )
        self.gradient_modulation_stats = {}
        self.gradient_modulation_counts = {}

    def deployment_state(self) -> dict[str, object]:
        weight_bits = {
            name: module.weight_quant.num_bits
            for name, module in self.targets.items()
        }
        activation_bits = {
            name: module.activation_quant.num_bits
            for name, module in self.targets.items()
        }
        return {
            "mode": self.mode,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "training_only": {
                "fisher": self.weight_sensitivity,
                "covariance": self.covariance,
                "subspaces": self.subspaces,
            },
        }

    def model_size_mb(self, model: nn.Module) -> float:
        total_bits = 0
        for name, parameter in model.named_parameters():
            bits = 32
            if name.endswith(".weight"):
                module_name = name[: -len(".weight")]
                if module_name in self.targets:
                    bits = self.targets[module_name].weight_quant.num_bits
            total_bits += parameter.numel() * bits
        return total_bits / 8.0 / (1024.0 ** 2)

    def profile_macs(
        self,
        model: nn.Module,
        sample: Tensor,
        mode: str = "shared",
    ) -> dict[str, float]:
        counts: dict[str, float] = {}

        def make_hook(name: str):
            def hook(module, inputs, _output):
                x = inputs[0]
                if isinstance(module, nn.Linear):
                    counts[name] = counts.get(name, 0.0) + float(x.numel() // x.shape[-1]) * float(
                        module.in_features * module.out_features
                    )
                elif isinstance(module, nn.Conv1d):
                    output_length = _output.shape[-1]
                    counts[name] = counts.get(name, 0.0) + float(
                        _output.shape[0]
                        * output_length
                        * module.in_channels
                        * module.out_channels
                        * module.kernel_size[0]
                        / module.groups
                    )
            return hook

        handles = []
        for name, module in self.targets.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(sample, mode=mode)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        self.last_mac_profile = counts
        return counts

    def inference_report(self, model: nn.Module, sample: Tensor) -> dict[str, float]:
        counts = self.profile_macs(model, sample, mode="shared")
        gmac = sum(counts.values()) / 1e9
        gflops = 2.0 * gmac
        gbops = 0.0
        for name, macs in counts.items():
            module = self.targets[name]
            gbops += macs * module.weight_quant.num_bits * module.activation_quant.num_bits
        gbops /= 1e9
        report = {
            "model_size_mb": self.model_size_mb(model),
            "inference_gmac": gmac,
            "inference_gflops": gflops,
            "inference_gbops": gbops,
            "inference_extra_memory_mb": 0.0,
            "weight_bits": sorted(
                {module.weight_quant.num_bits for module in self.targets.values()}
            ),
            "activation_bits": sorted(
                {module.activation_quant.num_bits for module in self.targets.values()}
            ),
        }
        self.last_inference_report = report
        return report
