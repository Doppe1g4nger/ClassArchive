"""Active example components: amplifier, mixer, and ADC."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..component import Component
from ..events import SignalPayload


class Amplifier(Component):
    """Apply linear gain and accumulate total gain in metadata.

    A low-noise amplifier (LNA) is just an ``Amplifier`` with a small delay.
    """

    def __init__(self, name: str, gain_db: float, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.gain_db = gain_db
        self._lin = 10 ** (gain_db / 20)

    def on_signal(self, payload: SignalPayload) -> SignalPayload:
        iq = (payload.iq * self._lin).astype(payload.iq.dtype)
        md = {**payload.metadata, "gain_db": payload.metadata.get("gain_db", 0.0) + self.gain_db}
        return replace(payload, iq=iq, metadata=md)


class Mixer(Component):
    """Frequency-shift the signal by multiplying with a local-oscillator tone.

    The output ``center_freq`` is shifted down by ``lo_freq`` (downconversion).
    """

    def __init__(self, name: str, lo_freq: float, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.lo_freq = lo_freq

    def on_signal(self, payload: SignalPayload) -> SignalPayload:
        n = np.arange(payload.iq.shape[-1])
        lo = np.exp(-2j * np.pi * self.lo_freq * n / payload.sample_rate).astype(payload.iq.dtype)
        iq = payload.iq * lo
        return replace(payload, iq=iq, center_freq=payload.center_freq - self.lo_freq)


class ADC(Component):
    """Quantize the IQ samples to a fixed number of bits over a full-scale range.

    Models the analog-to-digital boundary: real and imaginary parts are clipped
    to ``[-full_scale, full_scale]`` and rounded to ``2**bits`` levels.
    """

    def __init__(
        self,
        name: str,
        bits: int = 12,
        full_scale: float = 1.0,
        processing_delay: float = 0.0,
    ) -> None:
        super().__init__(name, processing_delay)
        self.bits = bits
        self.full_scale = full_scale
        self._levels = 2 ** bits
        self._step = (2 * full_scale) / self._levels

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -self.full_scale, self.full_scale)
        return np.round(clipped / self._step) * self._step

    def on_signal(self, payload: SignalPayload) -> SignalPayload:
        i = self._quantize(payload.iq.real)
        q = self._quantize(payload.iq.imag)
        iq = (i + 1j * q).astype(payload.iq.dtype)
        md = {**payload.metadata, "adc_bits": self.bits}
        return replace(payload, iq=iq, metadata=md)
