"""Transmitter components: the egress counterpart of ``signalRX``.

A :class:`Transmitter` outputs a signal buffer and hands it back to the
environment via :meth:`RFSystem.transmit <rfdes.system.RFSystem.transmit>`
instead of forwarding to in-graph subscribers. It supports two modes:

* **relay** -- fired by an incoming signal from upstream (e.g. a repeater); the
  default :meth:`on_signal` returns the (optionally transformed) buffer, which is
  then transmitted;
* **source** -- :meth:`fire` builds a buffer from a generator (or an explicit
  payload) and transmits it, e.g. a CW :class:`ToneTransmitter`.

The component must be added to an :class:`~rfdes.system.RFSystem` whose
``on_transmit`` hook is set.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from ..component import Component
from ..events import DataObject, SignalPayload
from ..state import PlatformState


class Transmitter(Component):
    """Send a signal buffer back to the environment.

    Args:
        name: Component name.
        processing_delay: Latency before the buffer reaches the environment
            (constant or callable, like any component).
        when_busy: Blocking policy for relay mode (``None``/``"queue"``/``"drop"``).
        generator: Optional ``generator(state) -> SignalPayload`` for source mode.
    """

    accepts = (SignalPayload,)
    produces = None  # output leaves the system; no in-graph subscribers
    _is_transmitter = True

    def __init__(
        self,
        name: str,
        processing_delay=0.0,
        when_busy: Optional[str] = None,
        generator: Optional[Callable[[Optional[PlatformState]], SignalPayload]] = None,
    ) -> None:
        super().__init__(name, processing_delay, when_busy)
        self.generator = generator

    def on_signal(self, data: SignalPayload) -> Optional[SignalPayload]:
        """Relay mode: transform an incoming buffer before transmit (default: pass)."""
        return data

    def _deliver(self, data: DataObject, delay: float) -> None:
        """Route the buffer to the environment after ``delay`` (not to subscribers)."""
        if self._scheduler is None:
            raise RuntimeError(
                f"transmitter {self.name!r} is not bound to a scheduler; "
                "add it to an RFSystem before running"
            )
        if self.system is None:
            raise RuntimeError(
                f"transmitter {self.name!r} is not attached to an RFSystem; "
                "add it with RFSystem.add() so it can reach the environment"
            )
        out = replace(data, start_time=data.start_time + delay)
        self._scheduler.schedule(
            delay,
            lambda p=out: self.system.transmit(p, source=self),
        )

    def fire(self, payload: Optional[SignalPayload] = None) -> None:
        """Source mode: emit a buffer now (built from ``payload`` or the generator).

        Only schedules events, so it may be called before or during a run.
        """
        if payload is None:
            if self.generator is None:
                raise RuntimeError(
                    f"transmitter {self.name!r}: fire() needs a payload or a generator"
                )
            state = self.system.state if self.system is not None else None
            payload = self.generator(state)
        self._emit(payload, trigger=payload)


class ToneTransmitter(Transmitter):
    """Source transmitter that emits a continuous-wave (CW) tone buffer."""

    def __init__(
        self,
        name: str,
        freq: float,
        sample_rate: float,
        num_samples: int,
        center_freq: float = 0.0,
        amplitude: float = 1.0,
        processing_delay=0.0,
    ) -> None:
        super().__init__(name, processing_delay=processing_delay)
        self.freq = freq
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.center_freq = center_freq
        self.amplitude = amplitude
        self.generator = self._generate

    def _generate(self, state: Optional[PlatformState]) -> SignalPayload:
        n = np.arange(self.num_samples)
        iq = (self.amplitude * np.exp(2j * np.pi * self.freq * n / self.sample_rate)).astype(
            np.complex64
        )
        now = self._scheduler.now() if self._scheduler is not None else 0.0
        return SignalPayload(
            iq=iq,
            sample_rate=self.sample_rate,
            center_freq=self.center_freq,
            start_time=now,
            metadata={"tx": self.name},
        )


class Repeater(Transmitter):
    """Relay transmitter that applies gain to an incoming buffer and retransmits."""

    def __init__(
        self,
        name: str,
        gain_db: float = 0.0,
        processing_delay=0.0,
        when_busy: Optional[str] = None,
    ) -> None:
        super().__init__(name, processing_delay=processing_delay, when_busy=when_busy)
        self.gain_db = gain_db
        self._lin = 10 ** (gain_db / 20)

    def on_signal(self, data: SignalPayload) -> SignalPayload:
        iq = (data.iq * self._lin).astype(data.iq.dtype)
        md = {**data.metadata, "gain_db": data.metadata.get("gain_db", 0.0) + self.gain_db,
              "tx": self.name}
        return replace(data, iq=iq, metadata=md)
