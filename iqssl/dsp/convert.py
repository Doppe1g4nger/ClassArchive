"""The single boundary between complex DSP and real-valued models.

Models never see complex tensors: ``conv1d`` rejects them outright, and complex
autograd buys us nothing here while adding a class of subtle bugs. So every
conversion funnels through these two functions, and they are the only place the
I/Q channel ordering is defined.
"""

from __future__ import annotations

import torch
from torch import Tensor


def ri_to_complex(x: Tensor) -> Tensor:
    """``(..., 2, L)`` real -> ``(..., L)`` complex64. Channel 0 is I, 1 is Q."""
    if x.shape[-2] != 2:
        raise ValueError(f"expected 2 channels (I, Q) at dim -2, got shape {tuple(x.shape)}")
    if x.is_complex():
        raise ValueError("input is already complex")
    return torch.complex(x[..., 0, :].float(), x[..., 1, :].float())


def complex_to_ri(x: Tensor) -> Tensor:
    """``(..., L)`` complex -> ``(..., 2, L)`` float32."""
    if not x.is_complex():
        raise ValueError(f"expected a complex tensor, got {x.dtype}")
    return torch.stack([x.real, x.imag], dim=-2).float()


def as_complex(x: Tensor) -> Tensor:
    """Accept either layout and return complex64. Convenience for op boundaries."""
    return x.to(torch.complex64) if x.is_complex() else ri_to_complex(x)


def wrap_complex(fn):  # type: ignore[no-untyped-def]
    """Decorator: let a complex-domain op accept and return ``(B, 2, L)`` real.

    Augmentation pipelines hand around real tensors; DSP ops want complex. This
    keeps the round-trip in exactly one place per op rather than at every call.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(x: Tensor, *args, **kwargs):  # type: ignore[no-untyped-def]
        was_real = not x.is_complex()
        out = fn(ri_to_complex(x) if was_real else x, *args, **kwargs)
        return complex_to_ri(out) if was_real else out

    return wrapper
