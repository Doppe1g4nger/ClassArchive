"""Event and payload data types for the RF discrete-event framework.

These are plain data carriers. They do not execute themselves; the
:class:`~rfdes.scheduler.Scheduler` invokes a callback that closes over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # avoid an import cycle; only needed for type hints
    from .component import Component

#: Default dtype for IQ buffers. complex64 matches typical SDR hardware and
#: halves memory relative to complex128; override per-payload when precision
#: matters.
DEFAULT_IQ_DTYPE = np.complex64

# The name used for the special event the RF environment model raises when it
# hands a buffer of IQ samples to the RF system.
SIGNAL_RX = "signalRX"


@dataclass(frozen=True)
class SignalPayload:
    """An immutable buffer of IQ samples flowing between components.

    Treat instances as read-only. Component transforms should return a *new*
    payload (e.g. via :func:`dataclasses.replace`) rather than mutating the
    ``iq`` array in place, so that fan-out to multiple subscribers cannot alias.

    Attributes:
        iq: Complex samples. Shape ``(N,)`` for a single channel or
            ``(channels, N)`` for multi-antenna / MIMO.
        sample_rate: Sample rate in Hz.
        center_freq: Center (carrier) frequency in Hz.
        start_time: Host-clock time of the first sample.
        metadata: Free-form accumulated state (e.g. cumulative gain, SNR, tags).
    """

    iq: np.ndarray
    sample_rate: float
    center_freq: float
    start_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        """Number of samples per channel."""
        return int(self.iq.shape[-1])


@dataclass(order=True)
class Event:
    """A scheduled delivery of a payload to a component.

    ``order=True`` plus the ``seq`` tie-breaker gives a deterministic, stable
    FIFO ordering when two events share a timestamp (``heapq`` is otherwise not
    stable on ties). Only ``time`` and ``seq`` participate in comparison.
    """

    time: float
    seq: int
    target: "Component" = field(compare=False)
    payload: SignalPayload = field(compare=False)
    kind: str = field(default=SIGNAL_RX, compare=False)
    extra: Any = field(default=None, compare=False)
