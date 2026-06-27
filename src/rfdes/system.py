"""The :class:`RFSystem` container and the ``signalRX`` host adapter.

``RFSystem`` holds the components, binds them to the host scheduler, and exposes
:meth:`RFSystem.on_signal_rx` -- the method the external RF-environment
simulator calls when it raises a ``signalRX`` event with a buffer of IQ samples.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .component import Component
from .events import DEFAULT_IQ_DTYPE, SignalPayload
from .scheduler import Scheduler


class RFSystem:
    """A composed RF system driven by an (external or reference) scheduler.

    Args:
        scheduler: The host event queue. Use
            :class:`~rfdes.scheduler.HeapScheduler` to run standalone, or an
            adapter around the external RF-environment simulator's queue.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.entry: Optional[Component] = None
        self._components: list[Component] = []

    def add(self, component: Component) -> Component:
        """Register ``component`` and bind it to the scheduler. Returns it."""
        component.bind(self.scheduler)
        self._components.append(component)
        return component

    def set_entry(self, component: Component) -> Component:
        """Designate the front-end component that receives ``signalRX`` buffers."""
        if component not in self._components:
            self.add(component)
        self.entry = component
        return component

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(self._components)

    def on_signal_rx(
        self,
        iq: np.ndarray,
        sample_rate: float,
        center_freq: float,
        t: Optional[float] = None,
        **metadata,
    ) -> None:
        """Handle a ``signalRX`` event from the RF environment model.

        The external simulator calls this with a fresh IQ buffer. Delivery to
        the entry component is scheduled at ``delay=0`` so it passes *through*
        the host queue, interleaving correctly with other events at the same
        timestamp rather than running ahead of them.

        Args:
            iq: Complex sample buffer (coerced to the framework default dtype if
                not already complex).
            sample_rate: Sample rate in Hz.
            center_freq: Center frequency in Hz.
            t: Host-clock time of the first sample; defaults to ``now()``.
            **metadata: Arbitrary tags carried with the payload.
        """
        if self.entry is None:
            raise RuntimeError("no entry component set; call set_entry() first")

        arr = np.asarray(iq)
        if not np.iscomplexobj(arr):
            arr = arr.astype(DEFAULT_IQ_DTYPE)

        payload = SignalPayload(
            iq=arr,
            sample_rate=sample_rate,
            center_freq=center_freq,
            start_time=self.scheduler.now() if t is None else t,
            metadata=dict(metadata),
        )
        entry = self.entry
        self.scheduler.schedule(0.0, lambda: entry.receive(payload))
