"""Scheduler abstraction and a standalone reference implementation.

The external RF-environment simulator owns the master clock and event queue.
Our framework is a *guest*: it places callbacks onto that queue. The
:class:`Scheduler` Protocol is the integration boundary -- adapt it to any host
DES engine. :class:`HeapScheduler` is a self-contained reference engine so the
framework is runnable and testable without the external simulator.
"""

from __future__ import annotations

import heapq
from typing import Any, Callable, Optional, Protocol, runtime_checkable


def labeled(callback: Callable[[], None], label: str) -> Callable[[], None]:
    """Tag a callback with a human-readable label for queue introspection.

    The label rides along as an attribute on the callable, so the framework can
    pass plain ``schedule(delay, callback)`` (keeping the :class:`Scheduler`
    Protocol minimal for external hosts) while :class:`HeapScheduler` can still
    surface a description in :meth:`HeapScheduler.pending`. Schedulers that don't
    introspect simply ignore it.
    """
    try:
        callback._rfdes_label = label  # type: ignore[attr-defined]
    except (AttributeError, TypeError):  # pragma: no cover - builtins etc.
        pass
    return callback


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
        # Each entry is (time, seq, callback, label). seq is unique, so heap
        # comparison never reaches callback/label.
        self._q: list[tuple[float, int, Callable[[], None], Optional[str]]] = []
        self._t: float = 0.0
        self._seq: int = 0
        self._cancelled: set[int] = set()

    def now(self) -> float:
        return self._t

    def schedule(
        self, delay: float, callback: Callable[[], None], label: Optional[str] = None
    ) -> int:
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay!r}")
        if label is None:
            label = getattr(callback, "_rfdes_label", None)
        handle = self._seq
        self._seq += 1
        heapq.heappush(self._q, (self._t + delay, handle, callback, label))
        return handle

    def cancel(self, handle: int) -> None:
        self._cancelled.add(handle)

    @property
    def empty(self) -> bool:
        """True if no live (non-cancelled) events remain."""
        return all(h in self._cancelled for _, h, _, _ in self._q)

    def step(self) -> bool:
        """Fire the next live event. Returns False if the queue is exhausted."""
        while self._q:
            t, handle, callback, _label = heapq.heappop(self._q)
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
            t, handle, callback, _label = self._q[0]
            if until is not None and t > until:
                break
            heapq.heappop(self._q)
            if handle in self._cancelled:
                self._cancelled.discard(handle)
                continue
            self._t = t
            callback()

    # -- introspection ----------------------------------------------------
    def pending(self) -> list[tuple[float, str]]:
        """Live (non-cancelled) events left on the queue as ``(time, label)``.

        Ordered exactly as they will fire (by time, then FIFO insertion). Use at
        any point during a run -- e.g. inside an event callback or after
        ``run(until=...)`` -- to inspect what remains.
        """
        live = sorted(e for e in self._q if e[1] not in self._cancelled)
        return [(t, label if label is not None else "event") for t, _seq, _cb, label in live]

    def format_queue(self) -> str:
        """A multi-line, human-readable dump of the pending events + timestamps."""
        events = self.pending()
        if not events:
            return f"[t={self._t:g}] queue empty (0 events pending)"
        lines = [f"[t={self._t:g}] {len(events)} event(s) pending:"]
        lines += [f"  t={t:g} : {label}" for t, label in events]
        return "\n".join(lines)

    def print_queue(self, file=None) -> None:
        """Print :meth:`format_queue` (defaults to stdout)."""
        print(self.format_queue(), file=file)
