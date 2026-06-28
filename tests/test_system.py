import numpy as np
import pytest

from rfdes import Component, HeapScheduler, RFSystem
from rfdes.components import ADC, Amplifier, Mixer, Recorder


def test_signal_rx_propagates_end_to_end():
    sched = HeapScheduler()
    system = RFSystem(sched)
    lna = system.add(Amplifier("lna", gain_db=20.0, processing_delay=1.0))
    mix = system.add(Mixer("mix", lo_freq=1e9, processing_delay=2.0))
    adc = system.add(ADC("adc", bits=12, full_scale=50.0, processing_delay=5.0))
    rec = system.add(Recorder("rec"))
    lna >> mix >> adc >> rec
    system.set_entry(lna)

    n = np.arange(64)
    iq = np.exp(2j * np.pi * 1e6 * n / 10e6).astype(np.complex64)
    system.on_signal_rx(iq, sample_rate=10e6, center_freq=2e9, t=0.0)
    sched.run()

    assert len(rec.payloads) == 1
    out = rec.payloads[0]
    # center freq shifted down by LO
    assert out.center_freq == pytest.approx(1e9)
    # cumulative gain recorded
    assert out.metadata["gain_db"] == pytest.approx(20.0)
    # end-to-end latency is the sum of per-component delays (signalRX at delay 0)
    assert sched.now() == pytest.approx(1.0 + 2.0 + 5.0)


def test_signal_rx_enters_through_queue_and_interleaves():
    """A signalRX handoff scheduled at delay 0 must interleave (not jump ahead)
    of other events already at t=now."""
    sched = HeapScheduler()
    system = RFSystem(sched)
    order = []

    class Marker(Component):
        def on_signal(self, payload):
            order.append("rx")
            return None

    system.set_entry(system.add(Marker("entry")))

    # Pre-queue a competing event at the same timestamp BEFORE the signalRX.
    sched.schedule(0.0, lambda: order.append("other"))
    system.on_signal_rx(np.zeros(4, dtype=np.complex64), 1e6, 1e9, t=0.0)
    sched.run()

    # FIFO at t=0: the pre-queued event fires first, proving rx went via queue.
    assert order == ["other", "rx"]


def test_real_input_coerced_to_complex():
    sched = HeapScheduler()
    system = RFSystem(sched)
    rec = system.add(Recorder("rec"))
    system.set_entry(rec)
    system.on_signal_rx(np.ones(4, dtype=np.float64), 1e6, 1e9)
    sched.run()
    assert np.iscomplexobj(rec.payloads[0].iq)


def test_on_signal_rx_without_entry_raises():
    system = RFSystem(HeapScheduler())
    with pytest.raises(RuntimeError):
        system.on_signal_rx(np.zeros(4, dtype=np.complex64), 1e6, 1e9)


def test_metadata_passed_through_signal_rx():
    sched = HeapScheduler()
    system = RFSystem(sched)
    rec = system.add(Recorder("rec"))
    system.set_entry(rec)
    system.on_signal_rx(np.zeros(4, dtype=np.complex64), 1e6, 1e9, source="env", snr=12.0)
    sched.run()
    assert rec.payloads[0].metadata["source"] == "env"
    assert rec.payloads[0].metadata["snr"] == 12.0


class TimeSink(Component):
    """Records the scheduler time at which each buffer is received."""

    produces = None

    def __init__(self, name, scheduler):
        super().__init__(name)
        self._sched = scheduler
        self.times = []

    def on_signal(self, data):
        self.times.append(self._sched.now())
        return None


def test_signal_rx_at_absolute_time():
    sched = HeapScheduler()
    system = RFSystem(sched)
    sink = system.add(TimeSink("sink", sched))
    system.set_entry(sink)
    system.on_signal_rx(np.ones(4, dtype=np.complex64), 1e6, 1e9, at=25.0)
    sched.run()
    assert sink.times == [25.0]                      # delivered at scheduler time 25
    assert sched.now() == 25.0


def test_signal_rx_default_start_time_is_delivery_time():
    sched = HeapScheduler()
    system = RFSystem(sched)
    rec = system.add(Recorder("rec"))
    system.set_entry(rec)
    system.on_signal_rx(np.ones(4, dtype=np.complex64), 1e6, 1e9, at=12.0)
    sched.run()
    assert rec.payloads[0].start_time == 12.0        # start_time defaults to `at`


def test_explicit_t_overrides_start_time_independent_of_at():
    sched = HeapScheduler()
    system = RFSystem(sched)
    rec = system.add(Recorder("rec"))
    system.set_entry(rec)
    system.on_signal_rx(np.ones(4, dtype=np.complex64), 1e6, 1e9, at=12.0, t=99.0)
    sched.run()
    assert rec.payloads[0].start_time == 99.0


def test_signal_rx_in_the_past_raises():
    sched = HeapScheduler()
    system = RFSystem(sched)
    sink = system.add(TimeSink("sink", sched))
    system.set_entry(sink)
    # advance the clock to t=10, then try to schedule at t=5
    sched.schedule(10.0, lambda: None)
    sched.run()
    with pytest.raises(ValueError):
        system.on_signal_rx(np.ones(4, dtype=np.complex64), 1e6, 1e9, at=5.0)


def test_multiple_signal_rx_fire_in_time_order():
    sched = HeapScheduler()
    system = RFSystem(sched)
    sink = system.add(TimeSink("sink", sched))
    system.set_entry(sink)
    iq = np.ones(4, dtype=np.complex64)
    system.on_signal_rx(iq, 1e6, 1e9, at=30.0)
    system.on_signal_rx(iq, 1e6, 1e9, at=10.0)
    system.on_signal_rx(iq, 1e6, 1e9, at=20.0)
    sched.run()
    assert sink.times == [10.0, 20.0, 30.0]
