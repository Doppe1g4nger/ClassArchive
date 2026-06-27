"""The :class:`Component` base class: a node in the RF system graph.

A component receives a :class:`~rfdes.events.SignalPayload`, runs the user's
transform (:meth:`Component.on_signal`), and then fans the result out to its
registered subscribers. Each emission is charged this component's
``processing_delay`` -- the latency the component adds is modeled by scheduling
the downstream delivery ``processing_delay`` later on the host queue.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .events import SignalPayload
from .scheduler import Scheduler


class Component:
    """Base class for RF system components.

    Subclasses override :meth:`on_signal` to model the component's behavior
    (gain, mixing, filtering, ...). The framework handles event scheduling,
    delay, and fan-out.

    Args:
        name: Human-readable identifier, used in diagnostics.
        processing_delay: Latency this component adds, in host-clock units.
            Charged on emit: outputs reach subscribers ``processing_delay``
            after this component receives its input.
    """

    def __init__(self, name: str, processing_delay: float = 0.0) -> None:
        if processing_delay < 0:
            raise ValueError("processing_delay must be non-negative")
        self.name = name
        self.processing_delay = processing_delay
        self._subscribers: list[Component] = []
        self._scheduler: Optional[Scheduler] = None

    # -- wiring -----------------------------------------------------------
    def bind(self, scheduler: Scheduler) -> None:
        """Attach the host scheduler. Called by :class:`~rfdes.system.RFSystem`."""
        self._scheduler = scheduler

    def subscribe(self, downstream: "Component") -> "Component":
        """Register ``downstream`` to receive this component's output.

        Returns ``downstream`` to allow fluent chaining::

            ant.subscribe(lna).subscribe(mixer).subscribe(adc)

        Call repeatedly for fan-out (one source, many subscribers).
        """
        self._subscribers.append(downstream)
        return downstream

    def __rshift__(self, downstream: "Component") -> "Component":
        """``a >> b`` is sugar for ``a.subscribe(b)``."""
        return self.subscribe(downstream)

    @property
    def subscribers(self) -> tuple["Component", ...]:
        return tuple(self._subscribers)

    # -- runtime ----------------------------------------------------------
    def receive(self, payload: SignalPayload) -> None:
        """Framework entry point, invoked by a scheduled callback.

        Runs the user transform and, if it returns a payload, fans it out.
        """
        result = self.on_signal(payload)
        if result is not None:
            self._emit(result)

    def on_signal(self, payload: SignalPayload) -> Optional[SignalPayload]:
        """Transform an incoming payload. Override in subclasses.

        Return a new :class:`~rfdes.events.SignalPayload` to forward downstream,
        or ``None`` to absorb the signal (e.g. a sink). The default is a
        pass-through.
        """
        return payload

    def _emit(self, payload: SignalPayload) -> None:
        """Schedule delivery of ``payload`` to each subscriber after the delay."""
        if self._scheduler is None:
            raise RuntimeError(
                f"component {self.name!r} is not bound to a scheduler; "
                "add it to an RFSystem (or call bind()) before running"
            )
        start_time = payload.start_time + self.processing_delay
        multi = len(self._subscribers) > 1
        for sub in self._subscribers:
            # Give each subscriber an independent payload so fan-out branches
            # cannot alias each other's IQ buffer. Skip the copy for a single
            # subscriber (the common case) to avoid needless allocation.
            iq = payload.iq.copy() if multi else payload.iq
            out = replace(payload, iq=iq, start_time=start_time)
            # default-arg binding pins the loop variables, avoiding the
            # late-binding closure bug.
            self._scheduler.schedule(
                self.processing_delay,
                lambda s=sub, p=out: s.receive(p),
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, delay={self.processing_delay})"
