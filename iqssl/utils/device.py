"""Device and precision resolution.

Two jobs, both of which exist because the failure they prevent is quiet rather
than loud.

**Devices.** ``torch.device("cuda")`` constructs happily on a machine with no
GPU and fails much later, somewhere inside a kernel launch, with a message that
does not mention the config key that asked for it. This resolves the request up
front and says which knob to change.

**Precision.** Mixed precision is worth 2-3x on a modern GPU and is the obvious
thing to reach for when moving this benchmark off CPU. It is also a *contract*
setting, not a performance flag: comparing a bf16 method against an fp32 one
measures numerical tolerance alongside objective quality. So precision lives in
``TrainConfig`` next to batch size and epochs, where ``METHOD_TUNABLE`` already
refuses to let a method config set it, rather than being auto-enabled per host.
"""

from __future__ import annotations

import torch

PRECISIONS = ("fp32", "bf16", "fp16")
"""Allowed values for ``TrainConfig.precision``."""


def resolve_device(name: str) -> torch.device:
    """Validate a device request, failing with a message that names the knob."""
    device = torch.device(name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "train.device=cuda but no CUDA device is visible. Set train.device=cpu, "
            "or train.device=mps on Apple silicon. A CPU fallback is deliberately "
            "not automatic: a run that silently drops to CPU looks identical in its "
            "output and takes ~100x longer to say so."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "train.device=mps but the MPS backend is unavailable (needs Apple "
            "silicon and a torch built with MPS). Set train.device=cpu."
        )
    return device


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """The dtype to run forward passes in, or None for full precision.

    Returns None on CPU whatever was asked for. CPU autocast exists but is bf16
    only, is slower than fp32 for these model sizes, and would mean the smoke
    runs in CI exercised a different numerical path than any real run -- which is
    the opposite of what the smoke tier is for.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; options: {list(PRECISIONS)}")
    if precision == "fp32" or device.type == "cpu":
        return None
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "train.precision=bf16 but this GPU does not support it (needs Ampere "
                "or newer). Use fp16, which needs the loss scaler, or fp32."
            )
        return torch.bfloat16
    return torch.float16


def needs_grad_scaler(dtype: torch.dtype | None) -> bool:
    """fp16 needs loss scaling to keep small gradients from flushing to zero.

    bf16 does not: it carries fp32's exponent range and trades mantissa bits
    instead, so nothing underflows and the scaler would only add a failure mode.
    """
    return dtype == torch.float16
