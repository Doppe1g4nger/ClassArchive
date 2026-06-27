"""Passive example components: attenuator, splitter, and a simple filter."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from ..component import Component
from ..events import SignalPayload


class Attenuator(Component):
    """Reduce signal amplitude by a fixed loss in dB."""

    def __init__(self, name: str, loss_db: float, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.loss_db = loss_db
        self._lin = 10 ** (-loss_db / 20)

    def on_signal(self, payload: SignalPayload) -> SignalPayload:
        iq = (payload.iq * self._lin).astype(payload.iq.dtype)
        md = {**payload.metadata, "gain_db": payload.metadata.get("gain_db", 0.0) - self.loss_db}
        return replace(payload, iq=iq, metadata=md)


class Splitter(Component):
    """Pass the signal through unchanged to every subscriber (power split).

    The framework already fans out to all subscribers; ``Splitter`` simply
    forwards the payload. Set ``loss_db`` to model insertion/split loss.
    """

    def __init__(self, name: str, loss_db: float = 0.0, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.loss_db = loss_db
        self._lin = 10 ** (-loss_db / 20)

    def on_signal(self, payload: SignalPayload) -> SignalPayload:
        if self.loss_db == 0.0:
            return payload
        iq = (payload.iq * self._lin).astype(payload.iq.dtype)
        return replace(payload, iq=iq)


class Filter(Component):
    """A brick-wall band-pass filter applied in the frequency domain.

    Frequencies outside ``[center_freq - bandwidth/2, center_freq + bandwidth/2]``
    (relative to the payload's center frequency / baseband) are zeroed.
    """

    def __init__(
        self,
        name: str,
        bandwidth: float,
        passband_center: float = 0.0,
        processing_delay: float = 0.0,
    ) -> None:
        super().__init__(name, processing_delay)
        self.bandwidth = bandwidth
        self.passband_center = passband_center

    def on_signal(self, payload: SignalPayload) -> Optional[SignalPayload]:
        n = payload.iq.shape[-1]
        if n == 0:
            return payload
        freqs = np.fft.fftfreq(n, d=1.0 / payload.sample_rate)
        mask = np.abs(freqs - self.passband_center) <= (self.bandwidth / 2)
        spectrum = np.fft.fft(payload.iq, axis=-1)
        spectrum = spectrum * mask
        iq = np.fft.ifft(spectrum, axis=-1).astype(payload.iq.dtype)
        return replace(payload, iq=iq)
