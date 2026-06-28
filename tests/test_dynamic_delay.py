import numpy as np
import pytest

from rfdes import (
    Component,
    HeapScheduler,
    MergeComponent,
    RFSystem,
    SignalPayload,
)
from rfdes.component import _PortRef
from rfdes.datatypes import PulseBuffer
from rfdes.delays import jitter, per_pulse, per_sample


def payload(n=8, val=1.0, start_time=0.0):
    return SignalPayload(np.full(n, val, dtype=np.complex64), 1e6, 1e9, start_time=start_time)


class TimeStampingSink(Component):
    accepts = (SignalPayload, PulseBuffer)
    produces = None

    def __init__(self, name, scheduler):
        super().__init__(name)
        self._sched = scheduler
        self.times = []

    def on_signal(self, data):
        self.times.append(self._sched.now())
        return None


# -- callable delay from input -----------------------------------------------
def test_callable_delay_uses_input():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src", processing_delay=lambda d: d.num_samples * 1.0))
    sink = system.add(TimeStampingSink("sink", sched))
    src.subscribe(sink)
    system.set_entry(src)

    system.on_signal_rx(payload(n=12).iq, 1e6, 1e9)
    sched.run()
    assert sink.times == [12.0]  # delay == num_samples of the input buffer


def test_constant_float_delay_still_works():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src", processing_delay=4.0))
    sink = system.add(TimeStampingSink("sink", sched))
    src.subscribe(sink)
    system.set_entry(src)
    system.on_signal_rx(payload().iq, 1e6, 1e9)
    sched.run()
    assert sink.times == [4.0]


def test_negative_callable_result_raises():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src", processing_delay=lambda d: -1.0))
    sink = system.add(TimeStampingSink("sink", sched))
    src.subscribe(sink)
    system.set_entry(src)
    system.on_signal_rx(payload().iq, 1e6, 1e9)
    with pytest.raises(ValueError):
        sched.run()


def test_fanout_resolves_delay_once():
    """Both branches of a fan-out must see the same (single) resolved delay."""
    calls = {"n": 0}

    def delay_fn(_data):
        calls["n"] += 1
        return 3.0

    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(Component("src", processing_delay=delay_fn))
    a = system.add(TimeStampingSink("a", sched))
    b = system.add(TimeStampingSink("b", sched))
    src.subscribe(a)
    src.subscribe(b)
    system.set_entry(src)
    system.on_signal_rx(payload().iq, 1e6, 1e9)
    sched.run()
    assert a.times == [3.0] and b.times == [3.0]
    assert calls["n"] == 1  # delay computed once per firing, not per subscriber


# -- delay factory helpers ----------------------------------------------------
def test_per_sample_factory():
    fn = per_sample(1.0, 0.5)
    assert fn(payload(n=10)) == pytest.approx(1.0 + 0.5 * 10)


def test_per_pulse_factory():
    fn = per_pulse(2.0, 0.25)
    pb = PulseBuffer(pulses=np.ones((4, 3)), sample_rate=1e6)
    assert fn(pb) == pytest.approx(2.0 + 0.25 * 4)


def test_jitter_is_deterministic_with_seed_and_clamped():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    f1 = jitter(5.0, 1.0, rng1)
    f2 = jitter(5.0, 1.0, rng2)
    vals1 = [f1(None) for _ in range(5)]
    vals2 = [f2(None) for _ in range(5)]
    assert vals1 == vals2  # same seed -> same sequence
    # clamp: a huge std around mean 0 can go negative; min_delay floors it
    rng3 = np.random.default_rng(0)
    clamped = jitter(0.0, 100.0, rng3, min_delay=0.0)
    assert all(clamped(None) >= 0.0 for _ in range(50))


def test_per_pulse_drives_component_timing():
    sched = HeapScheduler()
    system = RFSystem(sched)

    class PulseRelay(Component):
        accepts = (PulseBuffer,)
        produces = PulseBuffer

    relay = system.add(PulseRelay("relay", processing_delay=per_pulse(1.0, 2.0)))
    sink = system.add(TimeStampingSink("sink", sched))
    relay.subscribe(sink)

    pb = PulseBuffer(pulses=np.ones((3, 3)), sample_rate=1e6)
    relay.receive(pb)  # not via signalRX (input is a PulseBuffer)
    sched.run()
    assert sink.times == [1.0 + 2.0 * 3]  # base + per_pulse * 3


# -- merge delay receives the matched inputs dict ----------------------------
def test_merge_delay_callable_receives_inputs_dict():
    seen = {}

    class TwoIn(MergeComponent):
        inputs = {"a": (SignalPayload,), "b": (SignalPayload,)}
        produces = SignalPayload

        def on_merge(self, inputs):
            return inputs["a"]

    def delay_fn(inputs):
        seen["keys"] = set(inputs)
        return float(inputs["a"].num_samples)

    sched = HeapScheduler()
    system = RFSystem(sched)
    merge = system.add(TwoIn("merge", processing_delay=delay_fn))
    sink = system.add(TimeStampingSink("sink", sched))
    merge.subscribe(sink)

    merge.receive(payload(n=7), port="a")
    merge.receive(payload(n=7), port="b")
    sched.run()
    assert seen["keys"] == {"a", "b"}
    assert sink.times == [7.0]


# -- `>>` port references -----------------------------------------------------
def test_getitem_rejects_non_string_and_is_not_iterable():
    c = Component("c")
    with pytest.raises(TypeError):
        c[0]                       # non-string port index
    with pytest.raises(TypeError):
        iter(c)                    # components must not be (infinitely) iterable
    with pytest.raises(TypeError):
        list(c)


def test_getitem_returns_portref():
    class TwoIn(MergeComponent):
        inputs = {"x": (SignalPayload,), "y": (SignalPayload,)}
        produces = SignalPayload

        def on_merge(self, inputs):
            return inputs["x"]

    merge = TwoIn("m")
    ref = merge["x"]
    assert isinstance(ref, _PortRef)
    assert ref.component is merge and ref.port == "x"


def test_rshift_into_portref_wires_named_port_and_chains():
    class TwoIn(MergeComponent):
        inputs = {"x": (SignalPayload,), "y": (SignalPayload,)}
        produces = SignalPayload

        def on_merge(self, inputs):
            return inputs["x"]

    a = Component("a")
    b = Component("b")
    merge = TwoIn("m")
    rec_like = Component("r")

    result = a >> merge["x"]
    assert result is merge                      # chaining returns the merge
    b >> merge["y"]
    merge >> rec_like

    assert (merge, "x") in a.connections
    assert (merge, "y") in b.connections
    assert (rec_like, "in") in merge.connections
