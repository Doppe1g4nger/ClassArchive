import numpy as np
import pytest

from rfdes import Component, HeapScheduler, RFSystem, SignalPayload
from rfdes.components import Recorder


def make_payload(value=1.0, n=8, start_time=0.0):
    iq = np.full(n, value, dtype=np.complex64)
    return SignalPayload(iq=iq, sample_rate=1e6, center_freq=1e9, start_time=start_time)


class TimeStampingSink(Component):
    """Records the scheduler time at which each payload is received."""

    def __init__(self, name, scheduler):
        super().__init__(name)
        self._sched = scheduler
        self.receive_times = []

    def on_signal(self, payload):
        self.receive_times.append(self._sched.now())
        return None


def test_processing_delay_charged_on_emit():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src", processing_delay=5.0))
    sink = system.add(TimeStampingSink("sink", sched))
    src.subscribe(sink)
    system.set_entry(src)

    system.on_signal_rx(make_payload().iq, 1e6, 1e9)
    sched.run()

    # signalRX is delivered through the queue at the current clock (t=0, delay 0);
    # src then emits 5 later -> sink receives at scheduler time 5.
    assert sink.receive_times == [5.0]


def test_fanout_delivers_to_all_subscribers():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src", processing_delay=2.0))
    sinks = [system.add(TimeStampingSink(f"s{i}", sched)) for i in range(3)]
    for s in sinks:
        src.subscribe(s)
    system.set_entry(src)

    system.on_signal_rx(make_payload().iq, 1e6, 1e9, t=0.0)
    sched.run()

    for s in sinks:
        assert s.receive_times == [2.0]


def test_fanout_payloads_do_not_alias():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src"))
    a = system.add(Recorder("a"))
    b = system.add(Recorder("b"))
    src.subscribe(a)
    src.subscribe(b)
    system.set_entry(src)

    system.on_signal_rx(make_payload(value=1.0).iq, 1e6, 1e9)
    sched.run()

    # Mutating one branch's array must not affect the other's.
    a.payloads[0].iq[0] = 999
    assert b.payloads[0].iq[0] == pytest.approx(1.0)


def test_start_time_advances_by_delay():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src", processing_delay=3.0))
    rec = system.add(Recorder("rec"))
    src.subscribe(rec)
    system.set_entry(src)

    system.on_signal_rx(make_payload(start_time=0.0).iq, 1e6, 1e9, t=100.0)
    sched.run()
    assert rec.payloads[0].start_time == pytest.approx(103.0)


def test_subscribe_returns_downstream_for_chaining():
    a, b, c = Component("a"), Component("b"), Component("c")
    assert a.subscribe(b) is b
    assert (a >> b) is b
    # build a chain and verify topology
    Component("x").subscribe(b).subscribe(c)
    assert c in b.subscribers


def test_unbound_component_raises_on_emit():
    src = Component("orphan")
    src.subscribe(Component("down"))
    with pytest.raises(RuntimeError):
        src.receive(make_payload())


def test_negative_delay_rejected():
    with pytest.raises(ValueError):
        Component("bad", processing_delay=-1.0)
