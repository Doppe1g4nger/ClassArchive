"""Additional typed data objects that flow between components.

These complement :class:`~rfdes.events.SignalPayload` (raw IQ) with higher-level
products produced further down an RF chain: detected pulses, spectrograms, and a
generic fused detection report. Each is a frozen dataclass subclassing
:class:`~rfdes.events.DataObject`, so the pre-simulation type check can reason
about which components accept which products.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import DataObject


@dataclass(frozen=True)
class PulseBuffer(DataObject):
    """A set of detected pulses extracted from an IQ buffer.

    Attributes:
        pulses: Array of shape ``(num_pulses, 3)`` with rows
            ``[start_sample, width_samples, peak_amplitude]``.
        sample_rate: Sample rate of the source buffer, in Hz.
        start_time: Host-clock time of the source buffer's first sample.
        metadata: Free-form accumulated state.
    """

    pulses: np.ndarray
    sample_rate: float
    start_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def num_pulses(self) -> int:
        return int(self.pulses.shape[0])


@dataclass(frozen=True)
class Spectrogram(DataObject):
    """A time-frequency power representation of an IQ buffer.

    Attributes:
        power: Real array of shape ``(num_frames, num_bins)`` with per-bin power.
        freqs: Bin center frequencies (Hz), length ``num_bins``.
        times: Frame start times (host-clock units), length ``num_frames``.
        start_time: Host-clock time of the source buffer's first sample.
        metadata: Free-form accumulated state.
    """

    power: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    start_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.power.shape[0]), int(self.power.shape[1]))


@dataclass(frozen=True)
class DetectionReport(DataObject):
    """A fused, higher-level result combining several upstream products.

    Attributes:
        fields: Arbitrary named results (e.g. pulse count, peak frequency).
        start_time: Host-clock time the report pertains to.
        metadata: Free-form accumulated state.
    """

    fields: dict = field(default_factory=dict)
    start_time: float = 0.0
    metadata: dict = field(default_factory=dict)
