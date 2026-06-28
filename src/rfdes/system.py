"""The :class:`RFSystem` container and the ``signalRX`` host adapter.

``RFSystem`` holds the components, binds them to the host scheduler, and exposes
:meth:`RFSystem.on_signal_rx` -- the method the external RF-environment
simulator calls when it raises a ``signalRX`` event with a buffer of IQ samples.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .component import DEFAULT_PORT, Component, MergeComponent
from .events import DEFAULT_IQ_DTYPE, DataObject, SignalPayload
from .scheduler import Scheduler
from .state import PlatformState


class TypeCheckError(Exception):
    """Raised when an RF system's wiring carries incompatible data types.

    Aggregates every problem found during :meth:`RFSystem.validate` into a single
    message so the user can fix the whole graph at once.
    """


class RFSystem:
    """A composed RF system driven by an (external or reference) scheduler.

    Models a platform: it carries identity and 6DOF :class:`~rfdes.state.PlatformState`,
    receives signals from the environment via :meth:`on_signal_rx`, and transmits
    back via :meth:`transmit` (the egress used by
    :class:`~rfdes.components.transmitters.Transmitter`).

    Args:
        scheduler: The host event queue. Use
            :class:`~rfdes.scheduler.HeapScheduler` to run standalone, or an
            adapter around the external RF-environment simulator's queue.
        name: Platform name.
        state: Initial 6DOF platform state (defaults to a zero state named ``name``).
        on_transmit: Environment egress callback ``on_transmit(payload, state)``
            invoked when a transmitter sends a buffer back to the environment.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        name: str = "",
        state: Optional[PlatformState] = None,
        on_transmit: Optional[Callable[[DataObject, PlatformState], None]] = None,
    ) -> None:
        self.scheduler = scheduler
        self.name = name
        self.state = state if state is not None else PlatformState(name=name)
        if not self.state.name:
            self.state.name = name
        self.on_transmit = on_transmit
        self.entry: Optional[Component] = None
        self._components: list[Component] = []
        self._validated = False

    def add(self, component: Component) -> Component:
        """Register ``component``, bind its scheduler, and back-link it. Returns it."""
        component.bind(self.scheduler)
        component.system = self
        self._components.append(component)
        self._validated = False
        return component

    def set_entry(self, component: Component) -> Component:
        """Designate the front-end component that receives ``signalRX`` buffers."""
        if component not in self._components:
            self.add(component)
        self.entry = component
        self._validated = False
        return component

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(self._components)

    def validate(self) -> None:
        """Type-check the wiring; raise :class:`TypeCheckError` on any problem.

        Every connection is checked so that a producer's :attr:`Component.produces`
        type is accepted (by subclass) at the downstream port it feeds. Also
        verifies the entry accepts ``SignalPayload`` and that every merge port has
        a producer (an unfed merge port would never fire). Successful validation is
        cached; :meth:`add` / :meth:`set_entry` invalidate the cache.
        """
        errors: list[str] = []

        for comp in self._components:
            edges = comp.connections
            if edges and comp.produces is None:
                errors.append(
                    f"{comp.name!r} is a sink (produces nothing) but has "
                    f"{len(edges)} subscriber(s)"
                )
            for sub, port in edges:
                ports = sub.input_ports()
                if port not in ports:
                    errors.append(
                        f"{comp.name!r} -> {sub.name!r}: unknown input port "
                        f"{port!r} (known: {tuple(ports)})"
                    )
                    continue
                if comp.produces is None:
                    continue  # already reported above
                accepted = ports[port]
                if not issubclass(comp.produces, tuple(accepted)):
                    accepted_names = ", ".join(t.__name__ for t in accepted)
                    errors.append(
                        f"{comp.name!r} -> {sub.name!r} (port {port!r}): produces "
                        f"{comp.produces.__name__} but port accepts {accepted_names}"
                    )

        if self.entry is not None:
            entry_ports = self.entry.input_ports()
            accepted = entry_ports.get(DEFAULT_PORT, ())
            if not (accepted and issubclass(SignalPayload, tuple(accepted))):
                errors.append(
                    f"entry {self.entry.name!r} must accept SignalPayload on its "
                    f"{DEFAULT_PORT!r} port (signalRX delivers SignalPayload)"
                )

        fed: dict[int, set[str]] = {}
        for comp in self._components:
            for sub, port in comp.connections:
                fed.setdefault(id(sub), set()).add(port)
        for comp in self._components:
            if isinstance(comp, MergeComponent):
                missing = set(comp.inputs) - fed.get(id(comp), set())
                if missing:
                    errors.append(
                        f"merge {comp.name!r} has unfed input port(s) "
                        f"{sorted(missing)}; it would never fire"
                    )

        if self.on_transmit is None:
            transmitters = [c for c in self._components if getattr(c, "_is_transmitter", False)]
            if transmitters:
                names = ", ".join(repr(c.name) for c in transmitters)
                errors.append(
                    f"transmitter(s) {names} present but no on_transmit hook is set; "
                    "they could never deliver to the environment"
                )

        if errors:
            raise TypeCheckError(
                "RF system failed type validation:\n  - " + "\n  - ".join(errors)
            )
        self._validated = True

    def transmit(self, payload: DataObject, source: Optional[Component] = None) -> None:
        """Send a signal buffer back to the environment (the egress of signalRX).

        Called by a :class:`~rfdes.components.transmitters.Transmitter` when it
        fires. Schedules the ``on_transmit`` hook at ``delay=0`` so the handoff
        passes through the host queue (interleaving with other same-timestamp
        events), passing the payload and a snapshot of the platform 6DOF state so
        the environment can model propagation from this platform.
        """
        if self.on_transmit is None:
            raise RuntimeError(
                "no on_transmit hook set; cannot send a transmission to the "
                "environment (pass on_transmit=... to RFSystem)"
            )
        snapshot = self.state.snapshot()
        hook = self.on_transmit
        self.scheduler.schedule(0.0, lambda: hook(payload, snapshot))

    def on_signal_rx(
        self,
        iq: np.ndarray,
        sample_rate: float,
        center_freq: float,
        at: Optional[float] = None,
        t: Optional[float] = None,
        **metadata,
    ) -> None:
        """Handle a ``signalRX`` event from the RF environment model.

        The external simulator calls this with a fresh IQ buffer. The buffer is
        scheduled onto the host queue and delivered to the entry component at the
        chosen time, like any normal event, so it interleaves correctly with
        other same-timestamp events (FIFO).

        Args:
            iq: Complex sample buffer (coerced to the framework default dtype if
                not already complex).
            sample_rate: Sample rate in Hz.
            center_freq: Center frequency in Hz.
            at: Absolute scheduler time to deliver the buffer to the entry
                component; defaults to ``now()`` (immediate, ``delay=0``). Must
                not be in the past.
            t: Host-clock time of the first sample; defaults to the delivery time
                (``at`` or ``now()``).
            **metadata: Arbitrary tags carried with the payload.
        """
        if self.entry is None:
            raise RuntimeError("no entry component set; call set_entry() first")

        if not self._validated:
            self.validate()  # runtime type check before the first event runs

        now = self.scheduler.now()
        deliver_at = now if at is None else at
        delay = deliver_at - now
        if delay < 0:
            raise ValueError(
                f"cannot schedule signalRX in the past: at={at} is before now()={now}"
            )

        arr = np.asarray(iq)
        if not np.iscomplexobj(arr):
            arr = arr.astype(DEFAULT_IQ_DTYPE)

        payload = SignalPayload(
            iq=arr,
            sample_rate=sample_rate,
            center_freq=center_freq,
            start_time=deliver_at if t is None else t,
            metadata=dict(metadata),
        )
        entry = self.entry
        self.scheduler.schedule(delay, lambda: entry.receive(payload))
