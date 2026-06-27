"""The :class:`Component` base class: a node in the RF system graph.

A component receives a :class:`~rfdes.events.DataObject`, runs the user's
transform (:meth:`Component.on_signal`), and then fans the result out to its
registered subscribers. Each emission is charged this component's
``processing_delay`` -- the latency the component adds is modeled by scheduling
the downstream delivery ``processing_delay`` later on the host queue.

Components declare the data types they accept (:attr:`Component.accepts`) and the
type they produce (:attr:`Component.produces`) so that an RF system can be type
checked before simulation. :class:`MergeComponent` adds named input ports and
fires only once every port has received data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any, Callable, Optional, Union

from .events import DataObject, SignalPayload
from .scheduler import Scheduler

#: The implicit input port name used by ordinary single-input components.
DEFAULT_PORT = "in"

#: A processing delay: a constant, or a callable computing the delay from the
#: component's input (for ordinary components) or the matched ``{port: data}``
#: dict (for :class:`MergeComponent`).
DelaySpec = Union[float, Callable[[Any], float]]


class _PortRef:
    """A reference to a specific input port of a component.

    Produced by ``component[port]`` so a merge's named ports can be wired with
    the ``>>`` operator: ``producer >> merge["pulses"]``.
    """

    __slots__ = ("component", "port")

    def __init__(self, component: "Component", port: str) -> None:
        self.component = component
        self.port = port

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.component.name!r}[{self.port!r}]"


class Component:
    """Base class for RF system components.

    Subclasses override :meth:`on_signal` to model the component's behavior
    (gain, mixing, filtering, ...). The framework handles event scheduling,
    delay, and fan-out.

    Class attributes:
        accepts: Tuple of input data types this component accepts on its default
            port. A producer is compatible if its ``produces`` type is one of
            these or a subclass.
        produces: The data type this component emits, or ``None`` for a sink.

    Args:
        name: Human-readable identifier, used in diagnostics.
        processing_delay: Latency this component adds, in host-clock units.
            Charged on emit: outputs reach subscribers this many units after the
            component receives its input. May be a constant, or a callable that
            computes the delay from the component's input (e.g. delay growing
            with IQ length or pulse count, or random jitter around a mean).
    """

    accepts: tuple[type, ...] = (SignalPayload,)
    produces: Optional[type] = SignalPayload

    def __init__(self, name: str, processing_delay: DelaySpec = 0.0) -> None:
        if not callable(processing_delay) and processing_delay < 0:
            raise ValueError("processing_delay must be non-negative")
        self.name = name
        self.processing_delay = processing_delay
        # Each subscriber is stored with the downstream input port it feeds.
        self._subscribers: list[tuple[Component, str]] = []
        self._scheduler: Optional[Scheduler] = None

    # -- wiring -----------------------------------------------------------
    def bind(self, scheduler: Scheduler) -> None:
        """Attach the host scheduler. Called by :class:`~rfdes.system.RFSystem`."""
        self._scheduler = scheduler

    def subscribe(self, downstream: "Component", port: str = DEFAULT_PORT) -> "Component":
        """Register ``downstream`` to receive this component's output.

        Args:
            downstream: The component to feed.
            port: The downstream input port to deliver to. Ordinary components
                have a single ``"in"`` port; :class:`MergeComponent` subclasses
                expose named ports.

        Returns ``downstream`` to allow fluent chaining::

            ant.subscribe(lna).subscribe(mixer).subscribe(adc)

        Call repeatedly for fan-out (one source, many subscribers).
        """
        self._subscribers.append((downstream, port))
        return downstream

    def __rshift__(self, downstream: Union["Component", _PortRef]) -> "Component":
        """``a >> b`` is sugar for ``a.subscribe(b)``.

        ``b`` may be a component (delivered to its default port) or a port
        reference from ``component[port]`` (e.g. ``a >> merge["pulses"]``). Either
        way the downstream component is returned so chains can continue.
        """
        if isinstance(downstream, _PortRef):
            return self.subscribe(downstream.component, downstream.port)
        return self.subscribe(downstream)

    def __getitem__(self, port: str) -> _PortRef:
        """``component[port]`` -> a port reference usable with ``>>``."""
        return _PortRef(self, port)

    @property
    def subscribers(self) -> tuple["Component", ...]:
        """The downstream components this component feeds (ports omitted)."""
        return tuple(sub for sub, _ in self._subscribers)

    @property
    def connections(self) -> tuple[tuple["Component", str], ...]:
        """The ``(downstream, port)`` edges out of this component."""
        return tuple(self._subscribers)

    def input_ports(self) -> dict[str, tuple[type, ...]]:
        """Map of input port name -> accepted data types.

        Ordinary components expose a single default port. Override (as
        :class:`MergeComponent` does) for multiple named inputs.
        """
        return {DEFAULT_PORT: self.accepts}

    # -- runtime ----------------------------------------------------------
    def receive(self, data: DataObject, port: str = DEFAULT_PORT) -> None:
        """Framework entry point, invoked by a scheduled callback.

        Runs the user transform and, if it returns a data object, fans it out.
        ``port`` is accepted for interface uniformity but ignored by ordinary
        single-input components.
        """
        result = self.on_signal(data)
        if result is not None:
            self._emit(result, trigger=data)

    def on_signal(self, data: DataObject) -> Optional[DataObject]:
        """Transform an incoming data object. Override in subclasses.

        Return a new :class:`~rfdes.events.DataObject` to forward downstream, or
        ``None`` to absorb it (e.g. a sink). The default is a pass-through.
        """
        return data

    def _resolve_delay(self, trigger: Any) -> float:
        """Resolve this component's processing delay for one firing.

        ``trigger`` is the component's input (for ordinary components) or the
        matched ``{port: data}`` dict (for a merge); a callable
        ``processing_delay`` is invoked with it. Raises ``ValueError`` on a
        negative result.
        """
        spec = self.processing_delay
        delay = float(spec(trigger)) if callable(spec) else float(spec)
        if delay < 0:
            raise ValueError(
                f"component {self.name!r} produced a negative processing delay "
                f"({delay}); delays must be non-negative"
            )
        return delay

    def _emit(self, data: DataObject, trigger: Any = None) -> None:
        """Schedule delivery of ``data`` to each subscriber after the delay."""
        if self._scheduler is None:
            raise RuntimeError(
                f"component {self.name!r} is not bound to a scheduler; "
                "add it to an RFSystem (or call bind()) before running"
            )
        # Resolve the (possibly data-dependent or random) delay once per firing
        # so every fan-out branch sees the same, consistent latency.
        delay = self._resolve_delay(trigger if trigger is not None else data)
        out = replace(data, start_time=data.start_time + delay)
        multi = len(self._subscribers) > 1
        for sub, port in self._subscribers:
            # Give each subscriber an independent copy so fan-out branches cannot
            # alias each other's buffers. Skip the copy for a single subscriber
            # (the common case) to avoid needless allocation.
            payload = out.copy() if multi else out
            # default-arg binding pins the loop variables, avoiding the
            # late-binding closure bug.
            self._scheduler.schedule(
                delay,
                lambda s=sub, pt=port, p=payload: s.receive(p, pt),
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, delay={self.processing_delay})"


class MergeComponent(Component):
    """A component with multiple named input ports that joins their data.

    A merge fires only once **every** input port has pending data. By default it
    performs a FIFO *zip*: when all ports are non-empty it pops the oldest item
    from each and calls :meth:`on_merge` with a ``{port: data}`` mapping. Override
    :meth:`correlation_key` to instead match items that share a key (e.g. the
    originating signal's ``start_time``).

    Subclasses set:
        inputs: Map of port name -> accepted data types.
        produces: The fused output type.
    """

    #: Port name -> accepted data types. Subclasses must override.
    inputs: dict[str, tuple[type, ...]] = {}

    def __init__(self, name: str, processing_delay: DelaySpec = 0.0) -> None:
        super().__init__(name, processing_delay)
        if not self.inputs:
            raise ValueError(
                f"{type(self).__name__} must define a non-empty `inputs` mapping"
            )
        self._pending: dict[str, deque] = {port: deque() for port in self.inputs}

    def input_ports(self) -> dict[str, tuple[type, ...]]:
        return dict(self.inputs)

    def correlation_key(self, data: DataObject) -> Any:
        """Key used to match items across ports. ``None`` means FIFO (any item)."""
        return None

    def receive(self, data: DataObject, port: str = DEFAULT_PORT) -> None:
        if port not in self._pending:
            raise KeyError(
                f"{self.name!r} has no input port {port!r}; "
                f"known ports: {tuple(self._pending)}"
            )
        self._pending[port].append(data)
        group = self._find_ready_group()
        if group is not None:
            result = self.on_merge(group)
            if result is not None:
                # A merge's delay callable sees the matched {port: data} dict.
                self._emit(result, trigger=group)

    def _find_ready_group(self) -> Optional[dict[str, DataObject]]:
        """Return one item per port (and remove them) once a full set is ready."""
        # Not ready until every port has at least one pending item.
        if any(not self._pending[port] for port in self.inputs):
            return None

        keys_by_port = {
            port: [self.correlation_key(item) for item in self._pending[port]]
            for port in self.inputs
        }
        all_none = all(k is None for keys in keys_by_port.values() for k in keys)
        if all_none:
            # FIFO zip: take the oldest item from each port.
            return {port: self._pending[port].popleft() for port in self.inputs}

        # Key-based: find a key present in every port, then take the first match.
        common = set(keys_by_port[next(iter(self.inputs))])
        for port in self.inputs:
            common &= set(keys_by_port[port])
        common.discard(None)
        if not common:
            return None
        key = next(iter(common))
        group: dict[str, DataObject] = {}
        for port in self.inputs:
            dq = self._pending[port]
            # Remove by index (not value): data objects hold numpy arrays, so
            # equality-based removal would be ambiguous.
            chosen = next(i for i, item in enumerate(dq) if self.correlation_key(item) == key)
            group[port] = dq[chosen]
            del dq[chosen]
        return group

    def on_merge(self, inputs: dict[str, DataObject]) -> Optional[DataObject]:
        """Combine one item from each input port into an output. Override this.

        ``inputs`` maps each port name to its matched data object. Return the
        fused :class:`~rfdes.events.DataObject`, or ``None`` to absorb.
        """
        raise NotImplementedError
