"""Sink components that terminate a chain and capture results for inspection."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..component import Component
from ..events import SignalPayload


class Recorder(Component):
    """Capture every payload it receives. A terminal sink (forwards nothing).

    Inspect :attr:`records` after running the simulation. Each entry pairs the
    receive time (set by the caller, typically ``scheduler.now()``) with the
    payload, but the simplest use is to read ``recorder.payloads``.
    """

    def __init__(self, name: str, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.payloads: list[SignalPayload] = []

    def on_signal(self, payload: SignalPayload) -> Optional[SignalPayload]:
        self.payloads.append(payload)
        return None  # terminal sink


class SpectrumAnalyzer(Component):
    """Capture payloads and compute their power spectrum on demand. Terminal sink."""

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
        return np.abs(np.fft.fftshift(np.fft.fft(iq, axis=-1)))
