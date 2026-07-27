"""Power, SNR and additive noise.

The SNR convention is written down here once and everything else defers to it,
because an unstated SNR definition makes every "accuracy vs SNR" curve in the
thesis unfalsifiable.

**Definition.** ``SNR = E|s[n]|^2 / E|w[n]|^2`` with ``w ~ CN(0, sigma^2)``,
measured over the *full sampled band*. This matches the RadioML convention, so
numbers are comparable with the published literature.

The factor of two in :func:`add_awgn` is the classic off-by-two: a complex
sample carries ``sigma^2`` total noise power split evenly between I and Q, so
each real component gets ``sigma^2 / 2``. Getting this wrong yields an SNR that
is uniformly 3 dB off — large enough to matter, small enough to go unnoticed.

At ``sps`` samples per symbol the signal occupies roughly ``(1+beta)/sps`` of
the sampled band, so ``Es/N0`` runs about ``10*log10(sps/(1+beta))`` dB above
full-band SNR (~7.8 dB at sps=8, beta=0.25). :func:`es_n0_db` reports it, and
the generator stores both.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def signal_power(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Mean instantaneous power ``E|x|^2``."""
    p = x.real**2 + x.imag**2 if x.is_complex() else x**2
    return p.mean(dim=dim, keepdim=keepdim)


def rms(x: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    return signal_power(x, dim=dim, keepdim=keepdim).sqrt()


def add_awgn(
    x: Tensor,
    snr_db: Tensor | float,
    *,
    generator: torch.Generator | None = None,
    measured_power: Tensor | None = None,
    unit_noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Add complex AWGN at a target SNR. Returns ``(noisy, noise_power)``.

    ``measured_power`` overrides the signal power estimate. The generator passes
    the power of exactly the samples it will *keep*, since PA compression and
    the fading realization both shift power by several dB and the retained crop
    is what the SNR label must describe.

    ``unit_noise`` supplies externally-drawn unit-variance complex noise, so a
    generated sample depends only on its own RNG stream rather than on how the
    generation run happened to be batched.
    """
    if not x.is_complex():
        raise ValueError("add_awgn expects a complex input")

    snr_db_t = (
        snr_db.to(x.device, torch.float32)
        if isinstance(snr_db, Tensor)
        else torch.full((x.shape[0],), float(snr_db), device=x.device, dtype=torch.float32)
    )
    ps = signal_power(x) if measured_power is None else measured_power.to(x.device)
    snr_lin = torch.pow(10.0, snr_db_t / 10.0)
    noise_power = ps / snr_lin

    # sigma^2 total per complex sample => sigma^2 / 2 per real component.
    sigma_component = (noise_power / 2.0).sqrt().unsqueeze(-1)
    if unit_noise is None:
        nr = torch.randn(x.shape, generator=generator, device=x.device, dtype=torch.float32)
        ni = torch.randn(x.shape, generator=generator, device=x.device, dtype=torch.float32)
    else:
        nr, ni = unit_noise.real, unit_noise.imag
    noise = torch.complex(nr * sigma_component, ni * sigma_component)
    return x + noise, noise_power


def measure_snr_db(clean: Tensor, noisy: Tensor) -> Tensor:
    """Empirical SNR from a known clean reference. Used by tests, not training."""
    ps = signal_power(clean)
    pn = signal_power(noisy - clean)
    return 10.0 * torch.log10(ps / pn.clamp_min(1e-20))


def es_n0_db(snr_db: Tensor | float, sps: int, rolloff: float) -> Tensor | float:
    """Convert full-band SNR to per-symbol Es/N0.

    The signal occupies ``(1+rolloff)/sps`` of the sampled band, so the rest of
    the band contributes noise only.
    """
    delta = 10.0 * math.log10(sps / (1.0 + rolloff))
    return snr_db + delta


def normalize_rms(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale each buffer to unit RMS.

    Must be applied to the *noisy* buffer. Normalizing by clean-signal power
    would leave absolute received power intact as a shortcut feature perfectly
    correlated with SNR, and every robustness result would be contaminated.
    ``tests/test_dsp.py::test_normalization_no_snr_leak`` pins this.
    """
    scale = rms(x, keepdim=True).clamp_min(eps)
    return x / scale


def normalize_peak(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Scale each buffer to unit peak magnitude. Ablation alternative to RMS."""
    peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return x / peak


NORMALIZERS = {
    "rms": normalize_rms,
    "peak": normalize_peak,
    "none": lambda x: x,
}


def normalize(x: Tensor, kind: str = "rms") -> Tensor:
    try:
        return NORMALIZERS[kind](x)
    except KeyError:
        raise ValueError(
            f"unknown normalization {kind!r}; options: {sorted(NORMALIZERS)}"
        ) from None
