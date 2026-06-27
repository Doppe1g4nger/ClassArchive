import numpy as np
import pytest

from rfdes import Component, DataObject, HeapScheduler, RFSystem
from rfdes.components import PulseDetector, Recorder
from rfdes.datatypes import PulseBuffer, Spectrogram


def test_concrete_types_are_dataobjects():
    pb = PulseBuffer(pulses=np.zeros((0, 3)), sample_rate=1e6)
    sg = Spectrogram(power=np.zeros((2, 4)), freqs=np.zeros(4), times=np.zeros(2))
    assert isinstance(pb, DataObject)
    assert isinstance(sg, DataObject)


def test_copy_is_deep_and_independent():
    pb = PulseBuffer(pulses=np.ones((2, 3)), sample_rate=1e6)
    dup = pb.copy()
    dup.pulses[0, 0] = 999
    assert pb.pulses[0, 0] == 1.0  # original untouched


class PulsePassThrough(Component):
    """A non-IQ pass-through used to exercise the generic fan-out copy."""

    accepts = (PulseBuffer,)
    produces = PulseBuffer


def test_fanout_copy_generalizes_to_non_iq_types():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(PulsePassThrough("src"))
    a = system.add(Recorder("a"))
    b = system.add(Recorder("b"))
    src.subscribe(a)
    src.subscribe(b)

    pb = PulseBuffer(pulses=np.ones((1, 3)), sample_rate=1e6)
    src.receive(pb)
    sched.run()

    # Mutating one branch's PulseBuffer must not affect the other's.
    a.payloads[0].pulses[0, 0] = 42
    assert b.payloads[0].pulses[0, 0] == pytest.approx(1.0)


def test_start_time_advances_for_non_iq_types():
    sched = HeapScheduler()
    system = RFSystem(sched)
    src = system.add(PulsePassThrough("src", processing_delay=4.0))
    rec = system.add(Recorder("rec"))
    src.subscribe(rec)

    pb = PulseBuffer(pulses=np.zeros((0, 3)), sample_rate=1e6, start_time=10.0)
    src.receive(pb)
    sched.run()
    assert rec.payloads[0].start_time == pytest.approx(14.0)
