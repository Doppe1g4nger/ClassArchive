"""Demo: per-component blocking / queueing while processing.

A single slow component (processing one item at a time, busy for its
processing_delay) is fed a burst of arrivals spaced closer than it can process
them. We run it once with ``when_busy="queue"`` (late arrivals wait) and once
with ``when_busy="drop"`` (late arrivals are discarded), and print which items
made it through and when.

Run with::

    python examples/demo_blocking.py
"""

from __future__ import annotations

import numpy as np

from rfdes import Component, HeapScheduler, RFSystem


class TimeSink(Component):
    """Records the scheduler time at which each item arrives."""

    produces = None

    def __init__(self, name, scheduler):
        super().__init__(name)
        self._sched = scheduler
        self.times = []

    def on_signal(self, data):
        self.times.append(self._sched.now())
        return None


def run(policy: str) -> None:
    sched = HeapScheduler()
    system = RFSystem(sched, name="blocking-demo")
    slow = system.add(Component("slow", processing_delay=3.0, when_busy=policy))
    sink = system.add(TimeSink("sink", sched))
    slow.subscribe(sink)
    system.set_entry(slow)

    # Arrivals at t = 0,1,2,3,4 -- faster than the 3.0 processing time.
    iq = np.ones(8, dtype=np.complex64)
    for t in range(5):
        sched.schedule(float(t), lambda: system.on_signal_rx(iq, 1e6, 1e9))

    sched.run()
    print(f"when_busy={policy!r}:")
    print(f"  processed {len(sink.times)} of 5 arrivals at times {sink.times}")
    print(f"  dropped:  {slow.dropped}\n")


def main() -> None:
    print("Slow component (3.0 s/item) fed arrivals at t=0,1,2,3,4:\n")
    run("queue")
    run("drop")


if __name__ == "__main__":
    main()
