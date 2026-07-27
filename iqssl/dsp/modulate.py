"""Linear digital modulations: bits -> symbols -> RRC-shaped baseband.

Constellations are Gray-coded and normalized to unit average power, so a
"0 dB SNR" buffer means the same thing whether it carries BPSK or 64QAM.
Without that normalization, SNR would be entangled with modulation order and
every SNR-stratified result would be confounded.

Continuous-phase modulations (GFSK/CPFSK/MSK) are **not** here — they are not
``map -> upsample -> RRC`` and live in :mod:`iqssl.dsp.cpm`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from iqssl.dsp.filters import (
    DEFAULT_ROLLOFF,
    DEFAULT_SPAN,
    DEFAULT_SPS,
    filter_complex,
    rrc_taps,
    upsample,
)
from iqssl.registry import MODULATORS


def gray(i: int) -> int:
    """Binary-reflected Gray code. Adjacent constellation points differ by 1 bit,
    so a symbol error at moderate SNR usually costs one bit, not several."""
    return i ^ (i >> 1)


def _gray_pam_levels(m: int, device: torch.device | str = "cpu") -> Tensor:
    """``m``-level PAM amplitudes indexed by their Gray-coded symbol value."""
    amps = torch.arange(m, dtype=torch.float64, device=device) * 2 - (m - 1)
    out = torch.empty(m, dtype=torch.float64, device=device)
    for i in range(m):
        out[gray(i)] = amps[i]
    return out


def _unit_power(c: Tensor) -> Tensor:
    return c / torch.sqrt((c.real**2 + c.imag**2).mean())


@dataclass(frozen=True)
class ModSpec:
    """Static description of a modulation, independent of any signal."""

    name: str
    bits_per_symbol: int
    is_cpm: bool = False


class LinearModulator:
    """Base class for memoryless constellation modulations."""

    spec: ModSpec

    def constellation(self, device: torch.device | str = "cpu") -> Tensor:
        """``(M,)`` complex64, unit average power, indexed by symbol value."""
        raise NotImplementedError

    @property
    def order(self) -> int:
        return 1 << self.spec.bits_per_symbol

    def map_symbols(self, symbols: Tensor) -> Tensor:
        """``(B, N)`` int64 symbol values -> ``(B, N)`` complex64."""
        return self.constellation(symbols.device)[symbols]

    def demap(self, samples: Tensor) -> Tensor:
        """Nearest-constellation-point decision. ``(B, N)`` complex -> int64."""
        c = self.constellation(samples.device)
        d = (samples.unsqueeze(-1) - c.view(*([1] * samples.ndim), -1)).abs()
        return d.argmin(-1)

    def modulate(
        self,
        symbols: Tensor,
        *,
        sps: int = DEFAULT_SPS,
        rolloff: float = DEFAULT_ROLLOFF,
        span: int = DEFAULT_SPAN,
        mode: str = "full",
    ) -> Tensor:
        """Symbols -> pulse-shaped complex baseband.

        ``mode='full'`` keeps the filter transients so the caller can trim them
        deliberately; leaving a ramp-up at a fixed position is a shortcut
        feature a network will happily learn.
        """
        taps = rrc_taps(sps, rolloff, span, device=symbols.device)
        return filter_complex(upsample(self.map_symbols(symbols), sps), taps, mode=mode)


@MODULATORS.register("bpsk")
class BPSK(LinearModulator):
    spec = ModSpec("bpsk", 1)

    def constellation(self, device: torch.device | str = "cpu") -> Tensor:
        return torch.tensor([-1.0, 1.0], dtype=torch.complex64, device=device)


@MODULATORS.register("qpsk")
class QPSK(LinearModulator):
    spec = ModSpec("qpsk", 2)

    def constellation(self, device: torch.device | str = "cpu") -> Tensor:
        pam = _gray_pam_levels(2, device)
        c = pam.view(-1, 1) + 1j * pam.view(1, -1)
        return _unit_power(c.reshape(-1)).to(torch.complex64)


@MODULATORS.register("psk8")
class PSK8(LinearModulator):
    spec = ModSpec("psk8", 3)

    def constellation(self, device: torch.device | str = "cpu") -> Tensor:
        k = torch.arange(8, dtype=torch.float64, device=device)
        points = torch.polar(torch.ones_like(k), 2 * math.pi * k / 8)
        order = torch.tensor([gray(i) for i in range(8)], device=device)
        c = torch.zeros(8, dtype=torch.complex128, device=device)
        c[order] = points
        return c.to(torch.complex64)


class _SquareQAM(LinearModulator):
    """Gray-coded square QAM: the two halves of the bit-word index I and Q."""

    def constellation(self, device: torch.device | str = "cpu") -> Tensor:
        m = math.isqrt(self.order)
        pam = _gray_pam_levels(m, device)
        c = pam.view(-1, 1) + 1j * pam.view(1, -1)
        return _unit_power(c.reshape(-1)).to(torch.complex64)


@MODULATORS.register("qam16")
class QAM16(_SquareQAM):
    spec = ModSpec("qam16", 4)


@MODULATORS.register("qam64")
class QAM64(_SquareQAM):
    spec = ModSpec("qam64", 6)


@MODULATORS.register("pam4")
class PAM4(LinearModulator):
    spec = ModSpec("pam4", 2)

    def constellation(self, device: torch.device | str = "cpu") -> Tensor:
        return _unit_power(_gray_pam_levels(4, device).to(torch.complex128)).to(torch.complex64)


@MODULATORS.register("ook")
class OOK(LinearModulator):
    """On-off keying. Unit average power puts the 'on' level at sqrt(2).

    Worth keeping despite being the simplest: its amplitude envelope is exactly
    what MAE's per-patch target normalization would erase, which is why that
    option defaults off.
    """

    spec = ModSpec("ook", 1)

    def constellation(self, device: torch.device | str = "cpu") -> Tensor:
        return torch.tensor([0.0, math.sqrt(2.0)], dtype=torch.complex64, device=device)


LINEAR_MODULATIONS = ("bpsk", "qpsk", "psk8", "qam16", "qam64", "pam4", "ook")


def get_modulator(name: str) -> LinearModulator:
    from iqssl.dsp import cpm  # noqa: F401  -- ensure CPM classes are registered

    return MODULATORS.build(name)  # type: ignore[return-value]


def random_symbols(
    n_symbols: int,
    order: int,
    batch: int = 1,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    return torch.randint(0, order, (batch, n_symbols), generator=generator, device=device)


def symbols_to_bits(symbols: Tensor, bits_per_symbol: int) -> Tensor:
    """``(B, N)`` symbol values -> ``(B, N * bits_per_symbol)`` bits, MSB first."""
    shifts = torch.arange(bits_per_symbol - 1, -1, -1, device=symbols.device)
    bits = (symbols.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(symbols.shape[0], -1)


def bits_to_symbols(bits: Tensor, bits_per_symbol: int) -> Tensor:
    """Inverse of :func:`symbols_to_bits`."""
    b = bits.reshape(bits.shape[0], -1, bits_per_symbol)
    weights = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits.device)
    return (b * weights).sum(-1)


def matched_filter_downsample(
    x: Tensor,
    n_symbols: int,
    *,
    sps: int = DEFAULT_SPS,
    rolloff: float = DEFAULT_ROLLOFF,
    span: int = DEFAULT_SPAN,
) -> Tensor:
    """Receiver side of the clean path: matched filter, then sample at the
    symbol instants.

    Two ``mode='full'`` RRC convolutions place symbol ``k`` at sample
    ``span*sps + k*sps``, so the group delay is exact and no timing search is
    needed. Used by the zero-BER test, and available as an ablation — it is
    **off** in the generator, because matched filtering correlates the noise and
    would silently change what the stored SNR means.
    """
    taps = rrc_taps(sps, rolloff, span, device=x.device)
    y = filter_complex(x, taps, mode="full")
    start = span * sps
    idx = start + torch.arange(n_symbols, device=x.device) * sps
    return y[:, idx]
