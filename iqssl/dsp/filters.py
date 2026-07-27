"""Pulse-shaping filters and bandlimited interpolation.

Two things here are easy to get wrong and expensive to debug later.

**The RRC has two removable singularities**, at ``t = 0`` and ``t = ±T/(4β)``.
The closed form evaluates 0/0 at both. At the default ``sps=8, β=0.25`` the
second one lands on an *exact sample*, so a naive implementation does not
produce a slightly-wrong filter — it produces NaN, and every downstream buffer
becomes NaN. Both limits are evaluated analytically below.

**Interpolation is one primitive, not two.** A fixed fractional timing offset
and a continuous sample-rate offset (clock drift, in ppm) are the same
operation evaluated at different sample positions, so both route through
:func:`interpolate_at`. Notably *not* ``scipy.signal.resample``, which is
FFT-based and therefore wraps the end of the buffer onto the beginning — a
circular artifact that a network learns to exploit.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

DEFAULT_SPS = 8
DEFAULT_ROLLOFF = 0.25
DEFAULT_SPAN = 11
"""Odd, so the group delay ``span * sps / 2`` is a whole number of samples."""

INTERP_TAPS = 16
INTERP_KAISER_BETA = 8.0


def rrc_taps(
    sps: int = DEFAULT_SPS,
    rolloff: float = DEFAULT_ROLLOFF,
    span: int = DEFAULT_SPAN,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Root-raised-cosine impulse response, normalized to unit L2 energy.

    Returns ``(span * sps + 1,)`` taps, symmetric about the centre.
    """
    if not 0.0 < rolloff <= 1.0:
        raise ValueError(f"rolloff must be in (0, 1], got {rolloff}")
    if span % 2 == 0:
        raise ValueError(f"span should be odd so the delay is integral, got {span}")

    n = span * sps
    # float64 throughout: the singularity guards compare against an absolute
    # epsilon, and float32 rounding can place a sample just outside it.
    t = (torch.arange(n + 1, device=device, dtype=torch.float64) - n / 2) / sps
    beta = float(rolloff)
    eps = 1e-9

    num = torch.sin(math.pi * t * (1 - beta)) + 4 * beta * t * torch.cos(math.pi * t * (1 + beta))
    den = math.pi * t * (1 - (4 * beta * t) ** 2)

    # Fill with the generic form, then overwrite both removable singularities.
    h = torch.where(
        den.abs() > eps,
        num / torch.where(den.abs() > eps, den, torch.ones_like(den)),
        torch.zeros_like(den),
    )

    # t = 0:  h(0) = 1 + beta(4/pi - 1)
    at_zero = t.abs() < eps
    h = torch.where(at_zero, torch.full_like(h, 1.0 + beta * (4.0 / math.pi - 1.0)), h)

    # t = +/- T/(4 beta):  the (1 - (4 beta t)^2) factor vanishes.
    if beta > 0:
        t_sing = 1.0 / (4.0 * beta)
        at_sing = (t.abs() - t_sing).abs() < eps
        val = (beta / math.sqrt(2.0)) * (
            (1 + 2 / math.pi) * math.sin(math.pi / (4 * beta))
            + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta))
        )
        h = torch.where(at_sing, torch.full_like(h, val), h)

    if not torch.isfinite(h).all():
        raise RuntimeError("RRC taps contain non-finite values; singularity guard failed")

    h = h / torch.linalg.vector_norm(h)
    return h.to(dtype)


def rrc_singularity_values(rolloff: float) -> tuple[float, float]:
    """The two analytic limits, unnormalized. Exposed so tests can assert them
    against a numerical limit rather than against a copy of the same formula."""
    beta = float(rolloff)
    h0 = 1.0 + beta * (4.0 / math.pi - 1.0)
    hs = (beta / math.sqrt(2.0)) * (
        (1 + 2 / math.pi) * math.sin(math.pi / (4 * beta))
        + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta))
    )
    return h0, hs


def gaussian_taps(
    sps: int = DEFAULT_SPS,
    bt: float = 0.3,
    span: int = 3,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Gaussian frequency pulse for GFSK, normalized to unit *area*.

    Unit area, not unit energy: this shapes an instantaneous-frequency signal
    that is then integrated to phase, so the area sets the modulation index and
    must be preserved exactly.
    """
    n = span * sps
    t = (torch.arange(n + 1, device=device, dtype=torch.float64) - n / 2) / sps
    alpha = math.sqrt(math.log(2.0) / 2.0) / bt
    g = torch.exp(-(math.pi**2) * t**2 / (alpha**2)) * (math.sqrt(math.pi) / alpha)
    g = g / g.sum()
    return g.to(dtype)


def rect_taps(
    sps: int = DEFAULT_SPS,
    span: int = 1,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Rectangular (REC) frequency pulse for CPFSK/MSK, unit area."""
    n = span * sps
    g = torch.ones(n, device=device, dtype=torch.float64) / n
    return g.to(dtype)


def filter_complex(x: Tensor, taps: Tensor, mode: str = "same") -> Tensor:
    """Convolve a batch of complex signals with real taps.

    ``mode='full'`` keeps the transient tails (the generator needs them so it
    can trim deliberately); ``'same'`` centres the output on the input.
    """
    import torch.nn.functional as F

    if not x.is_complex():
        raise ValueError("filter_complex expects a complex input")
    b, ell = x.shape
    k = taps.numel()
    taps = taps.to(x.real.dtype)

    # Real and imaginary parts stack into the batch dim: one conv, not two.
    stacked = torch.cat([x.real, x.imag], dim=0).unsqueeze(1)  # (2B, 1, L)
    # conv1d cross-correlates; flip for true convolution.
    kernel = taps.flip(0).view(1, 1, k)
    pad = k - 1 if mode == "full" else k // 2
    out = F.conv1d(stacked, kernel, padding=pad)
    if mode == "same" and k % 2 == 0:
        out = out[..., :ell]
    real, imag = out.squeeze(1).split(b, dim=0)
    return torch.complex(real, imag)


def upsample(symbols: Tensor, sps: int) -> Tensor:
    """Zero-stuff by ``sps``: ``(B, N)`` -> ``(B, N * sps)``."""
    b, n = symbols.shape
    out = torch.zeros(b, n * sps, dtype=symbols.dtype, device=symbols.device)
    out[:, ::sps] = symbols
    return out


def _kaiser_sinc_taps(
    mu: Tensor, n_taps: int = INTERP_TAPS, beta: float = INTERP_KAISER_BETA
) -> Tensor:
    """Bandlimited interpolation taps for fractional offsets ``mu`` in [0, 1).

    Taps are computed *exactly* per requested offset rather than quantized into
    a subphase bank. A 32-phase bank would cap delay accuracy at 1/64 of a
    sample, which is the same order as the tolerance the tests assert; computing
    16 taps per position costs nothing by comparison.
    """
    k = n_taps // 2
    # offset j - K + 1 spans -(K-1) .. K, so (offset - mu) stays within [-K, K]
    # and the window argument stays in [-1, 1].
    offsets = torch.arange(n_taps, device=mu.device, dtype=mu.dtype) - (k - 1)
    delta = mu.unsqueeze(-1) - offsets  # (..., n_taps)

    sinc = torch.sinc(delta)
    arg = (delta / k).clamp(-1.0, 1.0)
    i0 = torch.special.i0
    window = i0(beta * torch.sqrt((1 - arg**2).clamp_min(0.0))) / i0(
        torch.tensor(beta, device=mu.device, dtype=mu.dtype)
    )

    taps = sinc * window
    return taps / taps.sum(-1, keepdim=True)  # unit DC gain: no amplitude ripple


def interpolate_at(x: Tensor, positions: Tensor, n_taps: int = INTERP_TAPS) -> Tensor:
    """Bandlimited interpolation of ``x`` at fractional sample ``positions``.

    ``x`` is ``(B, L)`` complex, ``positions`` is ``(B, M)`` in sample units.
    Returns ``(B, M)`` complex.

    This is the single primitive behind both fractional timing offset and
    sample-rate offset. Positions outside the buffer clamp to the edge, so
    boundary samples are not meaningful — tests compare interiors only.
    """
    if not x.is_complex():
        raise ValueError("interpolate_at expects a complex input")
    b, ell = x.shape
    k = n_taps // 2

    base = torch.floor(positions)
    mu = positions - base
    taps = _kaiser_sinc_taps(mu, n_taps)  # (B, M, n_taps)

    offsets = torch.arange(n_taps, device=x.device) - (k - 1)
    idx = base.long().unsqueeze(-1) + offsets  # (B, M, n_taps)
    idx = idx.clamp(0, ell - 1)

    flat = idx.reshape(b, -1)
    gathered_r = torch.gather(x.real, 1, flat).view_as(idx)
    gathered_i = torch.gather(x.imag, 1, flat).view_as(idx)

    taps = taps.to(x.real.dtype)
    return torch.complex((gathered_r * taps).sum(-1), (gathered_i * taps).sum(-1))
