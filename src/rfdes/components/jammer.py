"""Electronic-attack components: a pulse-driven jam controller.

:class:`JamController` is a merge that pairs a :class:`~rfdes.datatypes.DetectionReport`
with the originating RF buffer and decides whether to jam. It emits a barrage-noise
:class:`~rfdes.events.SignalPayload` (to be sent to the environment by a
:class:`~rfdes.components.transmitters.Transmitter`) only when pulses are present
*and* the carrier is at the target frequency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..component import MergeComponent
from ..datatypes import DetectionReport
from ..events import DataObject, SignalPayload


class JamController(MergeComponent):
    """Decide whether to jam, and synthesize the jamming waveform.

    Fires once it has both a detection ``report`` and the matching ``rf`` buffer.
    Produces a jamming :class:`~rfdes.events.SignalPayload` when
    ``report.fields["num_pulses"] >= min_pulses`` and the RF carrier is within
    ``freq_tol`` of ``target_freq``; otherwise produces nothing (no jam).

    Args:
        name: Component name.
        target_freq: Carrier to jam (Hz), default 2.4 GHz.
        freq_tol: Carrier match tolerance (Hz).
        min_pulses: Minimum detected pulses required to jam.
        amplitude: Jamming-noise amplitude.
        num_samples: Length of the jamming buffer (defaults to the RF buffer's).
        rng: ``numpy.random.Generator`` for reproducible noise.
        processing_delay: Jam-decision/preparation latency (constant or callable;
            a callable receives the matched ``{port: data}`` dict).
    """

    inputs = {"report": (DetectionReport,), "rf": (SignalPayload,)}
    produces = SignalPayload

    def __init__(
        self,
        name: str,
        target_freq: float = 2.4e9,
        freq_tol: float = 1e6,
        min_pulses: int = 1,
        amplitude: float = 1.0,
        num_samples: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        processing_delay=0.0,
    ) -> None:
        super().__init__(name, processing_delay)
        self.target_freq = target_freq
        self.freq_tol = freq_tol
        self.min_pulses = min_pulses
        self.amplitude = amplitude
        self.num_samples = num_samples
        self._rng = rng if rng is not None else np.random.default_rng()

    def on_merge(self, inputs: dict[str, DataObject]) -> Optional[SignalPayload]:
        report: DetectionReport = inputs["report"]  # type: ignore[assignment]
        rf: SignalPayload = inputs["rf"]  # type: ignore[assignment]

        num_pulses = int(report.fields.get("num_pulses", 0))
        on_target = abs(rf.center_freq - self.target_freq) <= self.freq_tol
        if num_pulses < self.min_pulses or not on_target:
            return None  # no jam

        n = self.num_samples if self.num_samples is not None else rf.num_samples
        noise = self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
        iq = (self.amplitude * noise / np.sqrt(2)).astype(np.complex64)
        return SignalPayload(
            iq=iq,
            sample_rate=rf.sample_rate,
            center_freq=self.target_freq,
            start_time=max(report.start_time, rf.start_time),
            metadata={"jam": True, "against_pulses": num_pulses},
        )
