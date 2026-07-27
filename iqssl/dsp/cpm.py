"""Continuous-phase modulations: GFSK, CPFSK, MSK.

These are **not** ``map -> upsample -> RRC``. They are phase-accumulation
modulations,

.. math:: \\phi(t) = 2\\pi h \\sum_k a_k\\, q(t - kT), \\qquad s(t) = e^{j\\phi(t)}

where ``q`` is the integral of a frequency pulse ``g``: rectangular (REC) for
CPFSK, Gaussian-filtered REC for GFSK, and MSK is exactly CPFSK at ``h=0.5``.

Two conventions worth stating because they are where the bugs live:

*Pulse area is 1/2, not 1.* The standard CPM normalization is
``q(inf) = 1/2``, giving a phase change of ``pi*h*a_k`` per symbol. With area 1
every modulation index would be doubled, and MSK would advance by ``pi`` per
symbol instead of ``pi/2``.

*No RRC afterwards.* Pulse-shaping a CPM signal destroys its constant modulus,
which is the entire point of the family and the reason it behaves so
differently from QAM under PA compression. ``test_cpm_constant_modulus``
exists to catch exactly that mistake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from iqssl.dsp.filters import DEFAULT_SPS, gaussian_taps, rect_taps, upsample
from iqssl.dsp.modulate import ModSpec
from iqssl.registry import MODULATORS

PULSE_AREA = 0.5
"""Standard CPM normalization: q(inf) = 1/2, so phase advances by pi*h per symbol."""


@dataclass(frozen=True)
class CPMConfig:
    h: float = 0.5
    pulse: str = "rec"
    pulse_span: int = 1
    bt: float = 0.3


class CPMModulator:
    """Base class for continuous-phase modulations."""

    spec: ModSpec
    config: CPMConfig

    @property
    def order(self) -> int:
        return 1 << self.spec.bits_per_symbol

    def frequency_pulse(self, sps: int, device: torch.device | str = "cpu") -> Tensor:
        cfg = self.config
        if cfg.pulse == "rec":
            g = rect_taps(sps, cfg.pulse_span, device=device)
        elif cfg.pulse == "gaussian":
            g = gaussian_taps(sps, cfg.bt, cfg.pulse_span, device=device)
        else:
            raise ValueError(f"unknown CPM pulse {cfg.pulse!r}; options: rec, gaussian")
        return g * PULSE_AREA

    def map_symbols(self, symbols: Tensor) -> Tensor:
        """Symbol values -> antipodal levels ``{-(M-1), ..., -1, 1, ..., M-1}``."""
        return symbols.to(torch.float32) * 2 - (self.order - 1)

    def phase(self, symbols: Tensor, *, sps: int = DEFAULT_SPS) -> Tensor:
        """Accumulated phase trajectory, ``(B, N * sps + span * sps - 1)``.

        Exposed separately from :meth:`modulate` so tests can assert continuity
        and the analytic per-symbol advance without going through ``exp``, where
        wrapping would hide a bug.
        """
        import torch.nn.functional as F

        levels = self.map_symbols(symbols)
        g = self.frequency_pulse(sps, symbols.device)
        impulses = upsample(levels, sps).unsqueeze(1)
        freq = F.conv1d(impulses, g.flip(0).view(1, 1, -1), padding=g.numel() - 1).squeeze(1)
        return 2.0 * math.pi * self.config.h * torch.cumsum(freq, dim=-1)

    def modulate(self, symbols: Tensor, *, sps: int = DEFAULT_SPS, **_: object) -> Tensor:
        """Symbols -> unit-modulus complex baseband.

        Accepts and ignores ``rolloff``/``span``/``mode`` so the generator can
        call every modulator through one interface; applying them here would be
        the bug this class exists to prevent.
        """
        phi = self.phase(symbols, sps=sps)
        return torch.polar(torch.ones_like(phi), phi).to(torch.complex64)

    def demod_differential(self, x: Tensor, n_symbols: int, *, sps: int = DEFAULT_SPS) -> Tensor:
        """Frequency-discriminator detection: the sign of the phase advance.

        Exact for full-response pulses (REC), where each symbol's phase change
        completes within its own interval. For partial-response pulses (Gaussian
        with span > 1) the pulses overlap by construction, so this detector has
        an irreducible ISI error floor and the tests do not demand zero BER.
        """
        idx = torch.arange(n_symbols + 1, device=x.device) * sps
        s = x[:, idx]
        dphi = torch.angle(s[:, 1:] * s[:, :-1].conj())
        return (dphi > 0).long()


@MODULATORS.register("cpfsk")
class CPFSK(CPMModulator):
    spec = ModSpec("cpfsk", 1, is_cpm=True)
    config = CPMConfig(h=0.5, pulse="rec", pulse_span=1)


@MODULATORS.register("msk")
class MSK(CPMModulator):
    """Minimum-shift keying: CPFSK at h=0.5 with a REC pulse.

    Kept as its own registry entry because it is a distinct label in the
    dataset, but ``test_msk_equals_cpfsk_h_half`` asserts the two produce
    identical samples — if they ever diverge, one of the configs drifted.
    """

    spec = ModSpec("msk", 1, is_cpm=True)
    config = CPMConfig(h=0.5, pulse="rec", pulse_span=1)


@MODULATORS.register("gfsk")
class GFSK(CPMModulator):
    """Gaussian FSK: partial-response, BT=0.3 over 3 symbols."""

    spec = ModSpec("gfsk", 1, is_cpm=True)
    config = CPMConfig(h=0.5, pulse="gaussian", pulse_span=3, bt=0.3)


CPM_MODULATIONS = ("gfsk", "cpfsk", "msk")
FULL_RESPONSE_CPM = ("cpfsk", "msk")
"""CPM schemes whose pulse fits in one symbol, so differential detection is exact."""
