import numpy as np
import pytest

from rfdes import (
    Component,
    ComponentTiming,
    HeapScheduler,
    MergeComponent,
    RFSystem,
    SignalPayload,
)
from rfdes.component import CONTROL_PORT, ControllableComponent
from rfdes.components import Recorder
from rfdes.datatypes import ControlMessage


class FakeClock:
    """A controllable stand-in for perf_counter."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def payload(n=4):
    return SignalPayload(np.ones(n, dtype=np.complex64), 1e6, 1e9)


def install_clock(comp, clock):
    comp._clock = clock
    return comp


def test_times_only_core_execution_not_message_passing():
    """The timed region covers on_signal only -- fan-out copies are excluded."""
    clock = FakeClock()

    class Work(Component):
        def on_signal(self, data):
            clock.t += 5.0          # 5 s of "core" work
            return data

    sched = HeapScheduler()
    system = RFSystem(sched)
    src = install_clock(system.add(Work("src")), clock)
    # two subscribers -> _deliver makes a copy per branch (message-passing work)
    system.add(Recorder("a"))
    system.add(Recorder("b"))
    src.subscribe(system.components[1])
    src.subscribe(system.components[2])
    system.set_entry(src)

    system.on_signal_rx(payload().iq, 1e6, 1e9)
    sched.run()

    assert src.exec_calls == 1
    assert src.exec_time == pytest.approx(5.0)     # exactly the on_signal ticks
    assert src.exec_last == pytest.approx(5.0)


def test_counts_per_call_and_mean():
    clock = FakeClock()

    class Work(Component):
        produces = None

        def on_signal(self, data):
            clock.t += 2.0
            return None

    sched = HeapScheduler()
    system = RFSystem(sched)
    sink = install_clock(system.add(Work("w")), clock)
    system.set_entry(sink)
    for _ in range(3):
        system.on_signal_rx(payload().iq, 1e6, 1e9)
    sched.run()

    assert sink.exec_calls == 3
    assert sink.exec_time == pytest.approx(6.0)
    assert sink.exec_mean == pytest.approx(2.0)


def test_timing_snapshot_and_reset():
    clock = FakeClock()

    class Work(Component):
        def on_signal(self, data):
            clock.t += 1.5
            return data

    w = install_clock(Work("w"), clock)
    HeapScheduler_bind(w)
    w.receive(payload())
    snap = w.timing()
    assert isinstance(snap, ComponentTiming)
    assert snap.name == "w" and snap.calls == 1 and snap.total == pytest.approx(1.5)

    w.reset_timing()
    assert w.exec_calls == 0 and w.exec_time == 0.0 and w.exec_last == 0.0


def HeapScheduler_bind(comp):
    comp.bind(HeapScheduler())


def test_merge_times_on_merge():
    clock = FakeClock()

    class Merge(MergeComponent):
        inputs = {"a": (SignalPayload,), "b": (SignalPayload,)}
        produces = SignalPayload

        def on_merge(self, inputs):
            clock.t += 4.0
            return inputs["a"]

    m = install_clock(Merge("m"), clock)
    m.bind(HeapScheduler())
    m.receive(payload(), port="a")
    assert m.exec_calls == 0             # not ready yet -> on_merge not called
    m.receive(payload(), port="b")
    assert m.exec_calls == 1
    assert m.exec_time == pytest.approx(4.0)


def test_control_execution_is_timed():
    clock = FakeClock()

    class Ctl(ControllableComponent):
        def on_control(self, msg):
            clock.t += 3.0

    c = install_clock(Ctl("c"), clock)
    c.bind(HeapScheduler())
    c.receive(ControlMessage(params={}), port=CONTROL_PORT)
    assert c.exec_calls == 1
    assert c.exec_time == pytest.approx(3.0)


def test_system_timing_report_sorted_and_resettable():
    clocks = {"fast": FakeClock(), "slow": FakeClock()}

    class Fast(Component):
        def on_signal(self, data):
            clocks["fast"].t += 1.0
            return data

    class Slow(Component):
        produces = None

        def on_signal(self, data):
            clocks["slow"].t += 9.0
            return None

    sched = HeapScheduler()
    system = RFSystem(sched)
    fast = install_clock(system.add(Fast("fast")), clocks["fast"])
    slow = install_clock(system.add(Slow("slow")), clocks["slow"])
    fast.subscribe(slow)
    system.set_entry(fast)

    system.on_signal_rx(payload().iq, 1e6, 1e9)
    sched.run()

    report = system.timing_report()
    assert [r.name for r in report] == ["slow", "fast"]   # sorted by total desc
    assert report[0].total == pytest.approx(9.0)
    assert report[1].total == pytest.approx(1.0)

    text = system.format_timing()
    assert "slow" in text and "component" in text

    system.reset_timing()
    assert all(r.total == 0.0 and r.calls == 0 for r in system.timing_report())
