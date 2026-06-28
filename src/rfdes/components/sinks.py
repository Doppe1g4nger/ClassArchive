"""Sink components that terminate a chain and capture results for inspection."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..component import Component
from ..events import DataObject, SignalPayload


class Recorder(Component):
    """Capture every data object it receives. A terminal sink (forwards nothing).

    Accepts any :class:`~rfdes.events.DataObject` so it can terminate chains of
    any data type. Inspect ``recorder.payloads`` after running the simulation.
    """

    accepts = (DataObject,)
    produces = None

    def __init__(self, name: str, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.payloads: list[DataObject] = []

    def on_signal(self, payload: DataObject) -> Optional[DataObject]:
        self.payloads.append(payload)
        return None  # terminal sink


class SpectrumAnalyzer(Component):
    """Capture payloads and compute their power spectrum on demand. Terminal sink."""

    accepts = (SignalPayload,)
    produces = None

    def __init__(self, name: str, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.payloads: list[SignalPayload] = []

    def on_signal(self, payload: SignalPayload) -> Optional[SignalPayload]:
        self.payloads.append(payload)
        return None

    def last_spectrum(self) -> Optional[np.ndarray]:
        """Magnitude spectrum (fft-shifted) of the most recent payload."""
        if not self.payloads:
            return None
        iq = self.payloads[-1].iq
        return np.abs(np.fft.fftshift(np.fft.fft(iq, axis=-1), axes=-1))
