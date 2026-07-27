"""Propagation channel: multipath fading, carrier and clock offsets.

These are the *nuisance* factors, resampled per buffer and — by construction —
statistically independent of the emitter parameters in :mod:`iqssl.dsp.impair`.
That independence is load-bearing for the thesis: if carrier frequency offset
partly encoded emitter identity, then "did the representation discard CFO?"
would have no interpretable answer, because discarding it would help one task
while destroying the other.

Fractional tap delays go through the same Kaiser-sinc interpolator as timing
offset and clock drift. Quantizing delays onto the sample grid would tie them to
a T/8 lattice at ``sps=8`` and correlate multipath structure with symbol timing,
manufacturing a shortcut that does not exist in real channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from iqssl.dsp.filters import INTERP_TAPS, _kaiser_sinc_taps, interpolate_at


@dataclass(frozen=True)
class TDLSpec:
    """Tap-delay-line channel description."""

    n_taps: int = 3
    delay_spread_symbols: float = 1.0
    k_factor_db: float = -100.0
    """Rician K. Very negative means pure Rayleigh (no specular component)."""
    doppler_hz: float = 0.0
    fading: str = "block"
    """``block`` holds the realization constant over the buffer (physically
    reasonable for 128 symbols); ``jakes`` varies it at ``doppler_hz``."""


def exponential_pdp(
    n_taps: int, decay_db_per_tap: float = 3.0, device: torch.device | str = "cpu"
) -> Tensor:
    """Exponentially decaying power delay profile, normalized so ``sum(p) = 1``.

    The normalization is what makes average channel gain unity, so fading does
    not covertly shift SNR.
    """
    l = torch.arange(n_taps, dtype=torch.float32, device=device)
    p = torch.pow(10.0, -decay_db_per_tap * l / 10.0)
    return p / p.sum()


def sample_taps(
    spec: TDLSpec,
    batch: int,
    sps: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Draw complex tap gains and fractional delays.

    Returns ``gains (B, n_taps)`` complex and ``delays (B, n_taps)`` in samples.

    Rayleigh taps use ``h_l = sqrt(p_l/2)(randn + j randn)`` so that
    ``E|h_l|^2 = p_l`` exactly. The Rician specular component is applied to the
    **line-of-sight tap only** — spreading ``K`` across every tap is a common
    bug that turns a Rician channel into something with no physical meaning.
    """
    n = spec.n_taps
    p = exponential_pdp(n, device=device)

    cn = torch.complex(
        torch.randn(batch, n, generator=generator, device=device),
        torch.randn(batch, n, generator=generator, device=device),
    ) / math.sqrt(2.0)  # E|cn|^2 = 1

    gains = torch.sqrt(p).unsqueeze(0) * cn

    if spec.k_factor_db > -30.0:
        k = 10.0 ** (spec.k_factor_db / 10.0)
        theta = torch.rand(batch, generator=generator, device=device) * 2 * math.pi
        los = torch.polar(torch.ones_like(theta), theta) * math.sqrt(k / (k + 1))
        diffuse = cn[:, 0] * math.sqrt(1.0 / (k + 1))
        gains[:, 0] = torch.sqrt(p[0]) * (los + diffuse)

    # Fractional, monotonically increasing delays across the spread.
    spread_samples = spec.delay_spread_symbols * sps
    if n > 1:
        u = torch.rand(batch, n, generator=generator, device=device)
        delays = torch.sort(u, dim=-1).values * spread_samples
        delays[:, 0] = 0.0  # first arrival defines the time reference
    else:
        delays = torch.zeros(batch, 1, device=device)

    return gains.to(torch.complex64), delays.to(torch.float32)


def build_cir(gains: Tensor, delays: Tensor, *, n_interp: int = INTERP_TAPS) -> Tensor:
    """Compose fractionally-delayed taps into one discrete impulse response.

    Each tap contributes a windowed-sinc kernel centred at its fractional delay;
    summing them gives a single ``(B, cir_len)`` filter, so applying the channel
    is one convolution rather than one per tap.
    """
    b = gains.shape[0]
    device = gains.device
    k = n_interp // 2

    base = torch.floor(delays)
    mu = delays - base
    taps = _kaiser_sinc_taps(mu, n_interp).to(torch.float32)  # (B, n_taps, n_interp)

    offsets = torch.arange(n_interp, device=device) - (k - 1)
    idx = base.long().unsqueeze(-1) + offsets  # (B, n_taps, n_interp)

    lo = int(idx.min().item())
    cir_len = int(idx.max().item()) - lo + 1
    cir = torch.zeros(b, cir_len, dtype=torch.complex64, device=device)

    contrib = gains.unsqueeze(-1) * taps  # (B, n_taps, n_interp)
    cir.scatter_add_(1, (idx - lo).reshape(b, -1), contrib.reshape(b, -1))
    return cir


def filter_per_sample(x: Tensor, taps: Tensor) -> Tensor:
    """Full convolution with a *different* complex kernel per batch element.

    Complex multiply expanded into four real grouped convolutions:
    ``(a+bi)(c+di) = (ac - bd) + i(ad + bc)``.
    """
    b, k = taps.shape

    def gconv(sig: Tensor, ker: Tensor) -> Tensor:
        return F.conv1d(
            sig.unsqueeze(0), ker.flip(-1).unsqueeze(1), padding=k - 1, groups=b
        ).squeeze(0)

    xr, xi = x.real.contiguous(), x.imag.contiguous()
    hr, hi = taps.real.contiguous(), taps.imag.contiguous()
    return torch.complex(gconv(xr, hr) - gconv(xi, hi), gconv(xr, hi) + gconv(xi, hr))


def apply_tdl(
    x: Tensor,
    spec: TDLSpec,
    sps: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a tap-delay-line channel. Returns ``(y, gains, delays)``.

    Output is longer than the input by the channel's span; the caller trims,
    because only the caller knows which samples it intends to keep.
    """
    gains, delays = sample_taps(spec, x.shape[0], sps, generator=generator, device=x.device)
    if spec.fading == "jakes" and spec.doppler_hz > 0:
        return apply_tdl_timevarying(x, gains, delays, spec, generator=generator), gains, delays
    return filter_per_sample(x, build_cir(gains, delays)), gains, delays


def jakes_envelope(
    batch: int,
    n_taps: int,
    length: int,
    doppler_norm: float,
    *,
    n_sinusoids: int = 8,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sum-of-sinusoids time-varying fading, ``(B, n_taps, L)``, unit mean power.

    A cheap Clarke/Jakes approximation: ``n_sinusoids`` plane waves with uniform
    arrival angles and random phases.
    """
    n = torch.arange(length, device=device, dtype=torch.float32)
    alpha = (
        torch.rand(batch, n_taps, n_sinusoids, 1, generator=generator, device=device) * 2 * math.pi
    )
    phase = (
        torch.rand(batch, n_taps, n_sinusoids, 1, generator=generator, device=device) * 2 * math.pi
    )
    arg = 2 * math.pi * doppler_norm * torch.cos(alpha) * n + phase
    env = torch.polar(torch.ones_like(arg), arg).sum(dim=2) / math.sqrt(n_sinusoids)
    return env.to(torch.complex64)


def apply_tdl_timevarying(
    x: Tensor,
    gains: Tensor,
    delays: Tensor,
    spec: TDLSpec,
    *,
    sample_rate_hz: float = 1e6,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Time-varying channel: each tap's gain evolves at the Doppler rate.

    Costs a gather over the whole buffer per tap, so it is off by default. Block
    fading is the physically reasonable default for 128-symbol buffers; this
    exists as a sweep axis for how methods handle non-stationarity.
    """
    b, ell = x.shape
    n_taps = gains.shape[1]
    env = jakes_envelope(
        b,
        n_taps,
        ell,
        spec.doppler_hz / sample_rate_hz,
        generator=generator,
        device=x.device,
    )
    tap_gains = gains.unsqueeze(-1) * env  # (B, n_taps, L)

    n = torch.arange(ell, device=x.device, dtype=torch.float32)
    out = torch.zeros_like(x)
    for l in range(n_taps):
        positions = (n.unsqueeze(0) - delays[:, l : l + 1]).clamp_min(0.0)
        out = out + tap_gains[:, l] * interpolate_at(x, positions)
    return out


def apply_cfo(
    x: Tensor,
    cfo_norm: Tensor | float,
    phase0: Tensor | float = 0.0,
) -> Tensor:
    """Carrier frequency offset. ``cfo_norm`` is cycles per sample."""
    b, ell = x.shape
    f = torch.as_tensor(cfo_norm, dtype=torch.float32, device=x.device)
    p0 = torch.as_tensor(phase0, dtype=torch.float32, device=x.device)
    if f.ndim == 0:
        f = f.expand(b)
    if p0.ndim == 0:
        p0 = p0.expand(b)
    n = torch.arange(ell, device=x.device, dtype=torch.float32)
    phase = 2 * math.pi * f.unsqueeze(-1) * n + p0.unsqueeze(-1)
    return x * torch.polar(torch.ones_like(phase), phase)


def apply_fractional_delay(x: Tensor, delay: Tensor | float) -> Tensor:
    """Static sub-sample timing offset. Positive ``delay`` shifts later in time."""
    b, ell = x.shape
    d = torch.as_tensor(delay, dtype=torch.float32, device=x.device)
    if d.ndim == 0:
        d = d.expand(b)
    n = torch.arange(ell, device=x.device, dtype=torch.float32)
    return interpolate_at(x, n.unsqueeze(0) - d.unsqueeze(-1))


def apply_sample_rate_offset(x: Tensor, ppm: Tensor | float, out_len: int | None = None) -> Tensor:
    """Continuous clock drift: resample at ``(1 + ppm*1e-6)`` times the rate.

    Deliberately not ``scipy.signal.resample``, which is FFT-based and wraps the
    buffer end onto its start. That circular artifact sits at a fixed position
    in every buffer, which is exactly the kind of shortcut a network finds.
    """
    b, ell = x.shape
    eps = torch.as_tensor(ppm, dtype=torch.float32, device=x.device) * 1e-6
    if eps.ndim == 0:
        eps = eps.expand(b)
    m = out_len if out_len is not None else ell
    n = torch.arange(m, device=x.device, dtype=torch.float32)
    positions = n.unsqueeze(0) * (1.0 + eps.unsqueeze(-1))
    return interpolate_at(x, positions.clamp(0, ell - 1))
