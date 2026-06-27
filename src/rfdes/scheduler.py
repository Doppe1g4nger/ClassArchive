"""Scheduler abstraction and a standalone reference implementation.

The external RF-environment simulator owns the master clock and event queue.
Our framework is a *guest*: it places callbacks onto that queue. The
:class:`Scheduler` Protocol is the integration boundary -- adapt it to any host
DES engine. :class:`HeapScheduler` is a self-contained reference engine so the
framework is runnable and testable without the external simulator.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class Scheduler(Protocol):
    """Minimal queue interface every DES engine can satisfy.

    Implement (or adapt) these three methods against the host simulator's
    queue. The ``schedule(delay, callback)`` shape is host-agnostic: SimPy
    (``env.timeout`` + a process), a bare ``heapq`` loop, and ``asyncio``
    (``call_later``) can all wrap it. The callback closes over the framework's
    own :class:`~rfdes.events.Event`; the host only needs to run it.
    """

    def now(self) -> float:
        """Return the current host-clock time."""
        ...

    def schedule(self, delay: float, callback: Callable[[], None]) -> Any:
        """Schedule ``callback`` to run at ``now() + delay``.

        Returns an opaque handle usable with :meth:`cancel`.
        """
        ...

    def cancel(self, handle: Any) -> None:
        """Cancel a previously scheduled callback. May be a no-op."""
        ...


class HeapScheduler:
    """A deterministic, standalone heap-based discrete-event scheduler.

    Events are ordered by ``(time, seq)`` so that ties break in FIFO order,
    matching :class:`~rfdes.events.Event`. Use this to run and test an RF system
    in isolation; in production, substitute the external simulator via the
    :class:`Scheduler` Protocol.
    """

    def __init__(self) -> None:
        self._q: list[tuple[float, int, Callable[[], None]]] = []
        self._t: float = 0.0
        self._seq: int = 0
        self._cancelled: set[int] = set()

    def now(self) -> float:
        return self._t

    def schedule(self, delay: float, callback: Callable[[], None]) -> int:
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay!r}")
        handle = self._seq
        self._seq += 1
        heapq.heappush(self._q, (self._t + delay, handle, callback))
        return handle

    def cancel(self, handle: int) -> None:
        self._cancelled.add(handle)

    @property
    def empty(self) -> bool:
        """True if no live (non-cancelled) events remain."""
        return all(h in self._cancelled for _, h, _ in self._q)

    def step(self) -> bool:
        """Fire the next live event. Returns False if the queue is exhausted."""
        while self._q:
            t, handle, callback = heapq.heappop(self._q)
            if handle in self._cancelled:
                self._cancelled.discard(handle)
                continue
            self._t = t
            callback()
            return True
        return False

    def run(self, until: float | None = None) -> None:
        """Run events in time order.

        If ``until`` is given, stop once the next event would fire after that
        time, leaving it (and later events) queued.
        """
        while self._q:
            t, handle, callback = self._q[0]
            if until is not None and t > until:
                break
            heapq.heappop(self._q)
            if handle in self._cancelled:
                self._cancelled.discard(handle)
                continue
            self._t = t
            callback()
