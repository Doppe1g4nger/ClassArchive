"""Detector / feature-extraction components and a merge (fusion) component.

These illustrate components that transform IQ into *different* data types
(:class:`~rfdes.datatypes.PulseBuffer`, :class:`~rfdes.datatypes.Spectrogram`)
and a :class:`~rfdes.component.MergeComponent` that fires only once it has
received both products and fuses them into a
:class:`~rfdes.datatypes.DetectionReport`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..component import MergeComponent
from ..component import Component
from ..datatypes import DetectionReport, PulseBuffer, Spectrogram
from ..events import DataObject, SignalPayload


class PulseDetector(Component):
    """Detect pulses where the IQ magnitude exceeds a threshold.

    Produces a :class:`~rfdes.datatypes.PulseBuffer` with one row per contiguous
    run of samples above ``threshold``: ``[start_sample, width, peak_amplitude]``.
    """

    accepts = (SignalPayload,)
    produces = PulseBuffer

    def __init__(self, name: str, threshold: float, processing_delay: float = 0.0) -> None:
        super().__init__(name, processing_delay)
        self.threshold = threshold

    def on_signal(self, payload: SignalPayload) -> PulseBuffer:
        mag = np.abs(payload.iq)
        if mag.ndim > 1:
            # Multi-channel (channels, N): detect on the per-sample envelope
            # (max magnitude across channels) so threshold tests stay scalar.
            mag = mag.max(axis=0)
        above = mag > self.threshold
        rows: list[list[float]] = []
        start: Optional[int] = None
        for i, hot in enumerate(above):
            if hot and start is None:
                start = i
            elif not hot and start is not None:
                rows.append([start, i - start, float(mag[start:i].max())])
                start = None
        if start is not None:
            rows.append([start, len(above) - start, float(mag[start:].max())])
        pulses = np.array(rows, dtype=float).reshape(-1, 3)
        return PulseBuffer(
            pulses=pulses,
            sample_rate=payload.sample_rate,
            start_time=payload.start_time,
            metadata={**payload.metadata, "threshold": self.threshold},
        )


class Spectrogrammer(Component):
    """Compute a spectrogram (framed FFT power) from an IQ buffer.

    Produces a :class:`~rfdes.datatypes.Spectrogram` of shape
    ``(num_frames, nfft)``.
    """

    accepts = (SignalPayload,)
    produces = Spectrogram

    def __init__(
        self,
        name: str,
        nfft: int = 64,
        hop: Optional[int] = None,
        processing_delay: float = 0.0,
    ) -> None:
        super().__init__(name, processing_delay)
        self.nfft = nfft
        self.hop = hop if hop is not None else nfft

    def on_signal(self, payload: SignalPayload) -> Spectrogram:
        iq = payload.iq
        n = iq.shape[-1]
        starts = list(range(0, max(n - self.nfft, 0) + 1, self.hop))
        frames = []
        for s in starts:
            frame = iq[..., s:s + self.nfft]
            spec = np.fft.fftshift(np.fft.fft(frame, n=self.nfft, axis=-1), axes=-1)
            power = np.abs(spec) ** 2
            if power.ndim > 1:
                # Multi-channel (channels, nfft): combine channels into one
                # total-power spectrum so the result stays (num_frames, nfft).
                power = power.sum(axis=0)
            frames.append(power)
        power = np.array(frames, dtype=float).reshape(-1, self.nfft)
        freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, d=1.0 / payload.sample_rate))
        times = payload.start_time + np.array(starts, dtype=float) / payload.sample_rate
        return Spectrogram(
            power=power,
            freqs=freqs,
            times=times,
            start_time=payload.start_time,
            metadata=dict(payload.metadata),
        )


class DetectionFusion(MergeComponent):
    """Fuse a pulse buffer and a spectrogram into a detection report.

    Fires only once it has received data on **both** input ports. Reports the
    pulse count and the spectrogram's peak-power frequency.
    """

    inputs = {"pulses": (PulseBuffer,), "spectrogram": (Spectrogram,)}
    produces = DetectionReport

    def on_merge(self, inputs: dict[str, DataObject]) -> DetectionReport:
        pulses: PulseBuffer = inputs["pulses"]  # type: ignore[assignment]
        spec: Spectrogram = inputs["spectrogram"]  # type: ignore[assignment]

        if spec.power.size:
            peak_idx = int(np.argmax(spec.power.sum(axis=0)))
            peak_freq = float(spec.freqs[peak_idx])
        else:
            peak_freq = float("nan")

        return DetectionReport(
            fields={
                "num_pulses": pulses.num_pulses,
                "peak_freq": peak_freq,
                "spectrogram_shape": spec.shape,
            },
            start_time=max(pulses.start_time, spec.start_time),
            metadata={},
        )
