import numpy as np
import pytest

from rfdes import Component, HeapScheduler, RFSystem, SignalPayload


class TimeSink(Component):
    produces = None

    def __init__(self, name, scheduler):
        super().__init__(name)
        self._sched = scheduler
        self.times = []

    def on_signal(self, data):
        self.times.append(self._sched.now())
        return None


def build(policy, delay=3.0):
    sched = HeapScheduler()
    system = RFSystem(sched)
    slow = system.add(Component("slow", processing_delay=delay, when_busy=policy))
    sink = system.add(TimeSink("sink", sched))
    slow.subscribe(sink)
    system.set_entry(slow)
    return sched, system, slow, sink


def feed_at_zero(system, n):
    iq = np.ones(4, dtype=np.complex64)
    for _ in range(n):
        system.on_signal_rx(iq, 1e6, 1e9)


def test_invalid_when_busy_rejected():
    with pytest.raises(ValueError):
        Component("c", when_busy="sometimes")


def test_queue_serializes_with_queueing_delay():
    sched, system, slow, sink = build("queue", delay=3.0)
    feed_at_zero(system, 2)  # both arrive at t=0
    sched.run()
    # first processed at t=3, second waited then processed at t=6
    assert sink.times == [3.0, 6.0]
    assert slow.dropped == 0


def test_queue_is_fifo_across_several():
    sched, system, slow, sink = build("queue", delay=2.0)
    feed_at_zero(system, 4)
    sched.run()
    assert sink.times == [2.0, 4.0, 6.0, 8.0]


def test_drop_discards_inputs_while_busy():
    sched, system, slow, sink = build("drop", delay=3.0)
    feed_at_zero(system, 3)  # first admitted, other two dropped
    sched.run()
    assert sink.times == [3.0]
    assert slow.dropped == 2


def test_processing_flag_toggles():
    sched, system, slow, sink = build("queue", delay=3.0)
    feed_at_zero(system, 1)
    assert slow.processing is False
    sched.step()  # fires the signalRX delivery -> slow begins processing
    assert slow.processing is True
    sched.run()   # release fires
    assert slow.processing is False


def test_non_blocking_default_allows_concurrency():
    sched, system, slow, sink = build(None, delay=3.0)
    feed_at_zero(system, 2)  # both at t=0
    sched.run()
    # no blocking: both processed concurrently, both emerge at t=3
    assert sink.times == [3.0, 3.0]
    assert slow.dropped == 0


def test_blocking_composes_with_dynamic_delay():
    # delay grows with buffer length; busy window uses that same per-item delay.
    sched = HeapScheduler()
    system = RFSystem(sched)
    slow = system.add(
        Component("slow", processing_delay=lambda d: float(d.num_samples), when_busy="queue")
    )
    sink = system.add(TimeSink("sink", sched))
    slow.subscribe(sink)
    system.set_entry(slow)

    a = np.ones(2, dtype=np.complex64)   # delay 2
    b = np.ones(5, dtype=np.complex64)   # delay 5
    system.on_signal_rx(a, 1e6, 1e9)
    system.on_signal_rx(b, 1e6, 1e9)
    sched.run()
    # first (2 samples) out at t=2; second (5 samples) starts at 2, out at 7
    assert sink.times == [2.0, 7.0]
