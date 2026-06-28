"""Demo: inspect the event queue at an arbitrary point in the run.

`HeapScheduler.print_queue()` (and `pending()`) dump the events and timestamps
still left to fire, labelled with what each event will do. Call it any time --
before running, after `run(until=...)`, or from inside a component.

Run with::

    python examples/demo_queue_inspect.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem
from rfdes.components import Amplifier, Recorder


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched, name="inspect-demo")

    lna = system.add(Amplifier("LNA", gain_db=10.0, processing_delay=2.0))
    rec_a = system.add(Recorder("recA"))
    rec_b = system.add(Recorder("recB"))
    lna.subscribe(rec_a)
    lna.subscribe(rec_b)
    system.set_entry(lna)

    # Queue three signalRX buffers at different absolute times.
    iq = np.ones(8, dtype=np.complex64)
    for t in (0.0, 10.0, 20.0):
        system.on_signal_rx(iq, 1e6, 1e9, at=t)

    print("Before running -- everything queued:")
    sched.print_queue()

    print("\nAfter run(until=0) -- first buffer fired, LNA fanned out:")
    sched.run(until=0.0)
    sched.print_queue()

    print("\nDraining the rest:")
    sched.run()
    sched.print_queue()


if __name__ == "__main__":
    main()
