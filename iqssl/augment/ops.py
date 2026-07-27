"""Individual view transforms, each wrapping a :mod:`iqssl.dsp` primitive.

Every op has the same shape: ``(B, 2, L)`` float32 in, ``(B, 2, L)`` float32 out,
with per-sample strengths drawn from an explicit :class:`torch.Generator`. Two
rules hold throughout and are worth stating once rather than at each call site.

**Draws come from the passed generator, never the global RNG.** Augmentation and
weight initialization must not share a stream, or changing the augmentation
policy would silently change model init too, and a policy ablation would
confound two variables at once.

**Ops are split into channel-side and hardware-side.** The distinction is the
whole argument of the benchmark, so it is structural rather than a comment: the
channel ops perturb nuisances the representation *should* discard, and the
hardware ops perturb the emitter fingerprint itself -- which is the label. The
latter exist only to build the ``hardware_invariant`` positive control.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Generator, Tensor

from iqssl.dsp import channel as ch
from iqssl.dsp import impair
from iqssl.dsp.convert import complex_to_ri, ri_to_complex
from iqssl.dsp.power import add_awgn, normalize_rms, signal_power
from iqssl.registry import AUG_OPS

Op = Callable[..., Tensor]


def _u(
    b: int, lo: float, hi: float, g: Generator | None, device: torch.device | str = "cpu"
) -> Tensor:
    """``(B,)`` uniform draws. One strength per sample, not one per batch."""
    return torch.rand(b, generator=g, device=device) * (hi - lo) + lo


def _complex(fn: Op) -> Op:
    """Adapt a complex-domain primitive to the real ``(B, 2, L)`` view layout."""

    def wrapper(x: Tensor, *args, **kwargs) -> Tensor:  # type: ignore[no-untyped-def]
        return complex_to_ri(fn(ri_to_complex(x), *args, **kwargs))

    wrapper.__name__ = fn.__name__
    return wrapper


# --- channel-side: nuisances a good representation should discard -------------


@AUG_OPS.register("cfo_jitter")
def cfo_jitter(x: Tensor, g: Generator | None = None, *, max_norm: float = 5e-4) -> Tensor:
    """Residual carrier frequency offset, in cycles per sample."""
    f = _u(x.shape[0], -max_norm, max_norm, g, x.device)
    return _complex(ch.apply_cfo)(x, f)


@AUG_OPS.register("phase_rotate")
def phase_rotate(x: Tensor, g: Generator | None = None) -> Tensor:
    """Uniform carrier phase. The cheapest and least controversial invariance:
    absolute phase is unobservable without a coherent reference."""
    theta = _u(x.shape[0], 0.0, 2 * math.pi, g, x.device)
    z = ri_to_complex(x) * torch.polar(torch.ones_like(theta), theta).unsqueeze(-1)
    return complex_to_ri(z)


@AUG_OPS.register("time_shift")
def time_shift(x: Tensor, g: Generator | None = None, *, max_samples: float = 4.0) -> Tensor:
    """Fractional timing offset via the bandlimited interpolator.

    Fractional rather than integer: a whole-sample roll would leave symbol timing
    on the same lattice in every view, so the model could stay sensitive to
    sub-sample timing and still solve the task.
    """
    d = _u(x.shape[0], -max_samples, max_samples, g, x.device)
    return _complex(ch.apply_fractional_delay)(x, d)


@AUG_OPS.register("sample_rate_offset")
def sample_rate_offset(x: Tensor, g: Generator | None = None, *, max_ppm: float = 30.0) -> Tensor:
    """Clock drift."""
    ppm = _u(x.shape[0], -max_ppm, max_ppm, g, x.device)
    return _complex(ch.apply_sample_rate_offset)(x, ppm)


@AUG_OPS.register("multipath")
def multipath(
    x: Tensor,
    g: Generator | None = None,
    *,
    n_taps: int = 2,
    delay_spread_symbols: float = 1.0,
    sps: int = 8,
) -> Tensor:
    """Draw a fresh fading realization and convolve.

    Output is trimmed back to the input length: views must stay the same shape,
    and the channel's span is not information the model should receive.
    """
    z = ri_to_complex(x)
    spec = ch.TDLSpec(n_taps=n_taps, delay_spread_symbols=delay_spread_symbols)
    gains, delays = ch.sample_taps(spec, z.shape[0], sps, generator=g, device=z.device)
    y = ch.filter_per_sample(z, ch.build_cir(gains, delays))
    return complex_to_ri(y[:, : z.shape[-1]])


@AUG_OPS.register("renoise")
def renoise(
    x: Tensor, g: Generator | None = None, *, snr_db: tuple[float, float] = (5.0, 25.0)
) -> Tensor:
    """Add further AWGN at a sampled SNR.

    The buffer already carries noise from generation, so this *lowers* effective
    SNR rather than setting it -- which is the honest framing: you cannot add
    negative noise, and an augmentation that pretended to set an absolute SNR
    would silently be a no-op on the noisiest half of the dataset.
    """
    z = ri_to_complex(x)
    snr = _u(z.shape[0], snr_db[0], snr_db[1], g, z.device)
    y, _ = add_awgn(z, snr, generator=g, measured_power=signal_power(z))
    return complex_to_ri(y)


# A per-buffer amplitude-scale op is deliberately absent. Every policy ends in
# `renormalize`, which divides the gain straight back out, so the op would be an
# exact identity dressed up as an augmentation -- and `none`, the one policy that
# does not renormalize, is meant to be the identity anyway. Scale invariance is
# already enforced upstream, by normalizing at generation.


@AUG_OPS.register("time_mask")
def time_mask(
    x: Tensor, g: Generator | None = None, *, max_frac: float = 0.15, n_masks: int = 1
) -> Tensor:
    """Zero contiguous spans, SpecAugment-style, in the time domain.

    Not a DSP effect and not physically motivated -- it is an occlusion prior,
    included because masked-input methods make one implicitly and a view-based
    method should be able to be given the same one for comparison.
    """
    b, _, ell = x.shape
    out = x.clone()
    span_max = max(1, int(ell * max_frac))
    for _ in range(n_masks):
        width = torch.randint(1, span_max + 1, (b,), generator=g, device=x.device)
        start = (torch.rand(b, generator=g, device=x.device) * (ell - width).clamp_min(1)).long()
        idx = torch.arange(ell, device=x.device).unsqueeze(0)
        span = (idx >= start.unsqueeze(-1)) & (idx < (start + width).unsqueeze(-1))
        out = out.masked_fill(span.unsqueeze(1), 0.0)
    return out


# --- hardware-side: these perturb the *label* ---------------------------------
#
# Only `hardware_invariant` uses them, and only as a positive control. An
# augmentation that randomizes IQ imbalance teaches the encoder to be invariant
# to IQ imbalance -- which is one of the things that distinguishes one emitter
# from another. Emitter accuracy is supposed to collapse; if it does not, the
# generator is not putting the fingerprint where it claims to.


@AUG_OPS.register("iq_imbalance_jitter")
def iq_imbalance_jitter(
    x: Tensor, g: Generator | None = None, *, max_gain_db: float = 1.5, max_phase_deg: float = 10.0
) -> Tensor:
    b = x.shape[0]
    return _complex(impair.iq_imbalance)(
        x,
        _u(b, -max_gain_db, max_gain_db, g, x.device),
        _u(b, -max_phase_deg, max_phase_deg, g, x.device),
    )


@AUG_OPS.register("dc_jitter")
def dc_jitter(
    x: Tensor, g: Generator | None = None, *, dbc: tuple[float, float] = (-32.0, -18.0)
) -> Tensor:
    b = x.shape[0]
    return _complex(impair.dc_offset)(
        x, _u(b, dbc[0], dbc[1], g, x.device), _u(b, dbc[0], dbc[1], g, x.device), dbc=True
    )


@AUG_OPS.register("pa_jitter")
def pa_jitter(
    x: Tensor, g: Generator | None = None, *, ibo_db: tuple[float, float] = (3.0, 12.0)
) -> Tensor:
    return _complex(impair.saleh)(x, ibo_db=_u(x.shape[0], ibo_db[0], ibo_db[1], g, x.device))


# --- shared tail --------------------------------------------------------------


def renormalize(x: Tensor) -> Tensor:
    """Restore unit power after a chain of ops.

    Applied at the end of every policy except ``none``. Several ops change
    absolute power -- the PA compresses, multipath fades, added noise raises it --
    and leaving that in would let the model read off *which augmentations were
    applied* from the buffer's gain alone, turning the view distribution into a
    label.
    """
    return complex_to_ri(normalize_rms(ri_to_complex(x)))
