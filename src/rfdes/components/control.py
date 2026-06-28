"""Closed-loop feedback components: a tunable filter and a scan scheduler.

`TunableBandpassFilter` is a :class:`~rfdes.component.ControllableComponent` whose
passband can be retuned at runtime via a control message. `ScanScheduler` watches
the pulse detector's output and feeds control messages back to the filter,
forming a closed loop that scans frequency bands until it finds a signal.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from ..component import Component, ControllableComponent
from ..datatypes import ControlMessage, PulseBuffer
from ..events import DataObject, SignalPayload


class TunableBandpassFilter(ControllableComponent):
    """Brick-wall band-pass filter whose passband is retunable via control.

    Frequencies outside ``[passband_center +/- bandwidth/2]`` (baseband) are
    zeroed. A :class:`~rfdes.datatypes.ControlMessage` on the ``control`` port
    updates ``passband_center`` and/or ``bandwidth`` for subsequent buffers.
    """

    accepts = (SignalPayload,)
    produces = SignalPayload

    def __init__(
        self,
        name: str,
        bandwidth: float,
        passband_center: float = 0.0,
        processing_delay=0.0,
        when_busy: Optional[str] = None,
    ) -> None:
        super().__init__(name, processing_delay, when_busy)
        self.bandwidth = bandwidth
        self.passband_center = passband_center

    def on_signal(self, payload: SignalPayload) -> SignalPayload:
        n = payload.iq.shape[-1]
        if n == 0:
            return payload
        freqs = np.fft.fftfreq(n, d=1.0 / payload.sample_rate)
        mask = np.abs(freqs - self.passband_center) <= (self.bandwidth / 2)
        spectrum = np.fft.fft(payload.iq, axis=-1) * mask
        iq = np.fft.ifft(spectrum, axis=-1).astype(payload.iq.dtype)
        return replace(payload, iq=iq)

    def on_control(self, msg: DataObject) -> None:
        params = getattr(msg, "params", {})
        if "passband_center" in params:
            self.passband_center = float(params["passband_center"])
        if "bandwidth" in params:
            self.bandwidth = float(params["bandwidth"])


class ScanScheduler(Component):
    """Closed-loop scan controller for a :class:`TunableBandpassFilter`.

    Watches a :class:`~rfdes.datatypes.PulseBuffer` from the detector. If pulses
    were seen it dwells (no retune); otherwise it advances to the next band and
    emits a :class:`~rfdes.datatypes.ControlMessage` retuning the filter for the
    next buffer. Wire the output to ``filter["control"]``.

    Args:
        name: Component name.
        bands: Passband-center frequencies to scan through (baseband, Hz).
        bandwidth: Passband width to command at each band.
        start_index: Index of the band assumed active initially.
        min_pulses: Pulses needed to count as a detection (dwell).
        processing_delay: Decision latency.
    """

    accepts = (PulseBuffer,)
    produces = ControlMessage

    def __init__(
        self,
        name: str,
        bands: list[float],
        bandwidth: float,
        start_index: int = 0,
        min_pulses: int = 1,
        processing_delay=0.0,
    ) -> None:
        super().__init__(name, processing_delay)
        if not bands:
            raise ValueError("ScanScheduler needs at least one band")
        self.bands = list(bands)
        self.bandwidth = bandwidth
        self.index = start_index % len(self.bands)
        self.min_pulses = min_pulses
        self.locked = False

    @property
    def current_band(self) -> float:
        return self.bands[self.index]

    def on_signal(self, pulses: PulseBuffer) -> Optional[ControlMessage]:
        if pulses.num_pulses >= self.min_pulses:
            self.locked = True
            return None  # dwell on the current band
        # No detection: advance to the next band and command the retune.
        self.locked = False
        self.index = (self.index + 1) % len(self.bands)
        return ControlMessage(
            params={"passband_center": self.bands[self.index], "bandwidth": self.bandwidth},
            start_time=pulses.start_time,
            metadata={"scan_band_index": self.index},
        )
