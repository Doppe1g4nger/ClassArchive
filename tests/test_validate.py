import numpy as np
import pytest

from rfdes import Component, HeapScheduler, RFSystem, SignalPayload, TypeCheckError
from rfdes.components import (
    Amplifier,
    DetectionFusion,
    PulseDetector,
    Recorder,
    Spectrogrammer,
    Splitter,
)


def build_system():
    sched = HeapScheduler()
    system = RFSystem(sched)
    return sched, system


def test_valid_graph_passes():
    _, system = build_system()
    lna = system.add(Amplifier("lna", gain_db=10.0))
    split = system.add(Splitter("split"))
    pd = system.add(PulseDetector("pd", threshold=0.5))
    sp = system.add(Spectrogrammer("sp"))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("rec"))
    lna.subscribe(split)
    split.subscribe(pd)
    split.subscribe(sp)
    pd.subscribe(fusion, port="pulses")
    sp.subscribe(fusion, port="spectrogram")
    fusion.subscribe(rec)
    system.set_entry(lna)
    system.validate()  # should not raise


def test_type_mismatch_raises_and_names_edge():
    _, system = build_system()
    lna = system.add(Amplifier("lna", gain_db=10.0))
    sp = system.add(Spectrogrammer("sp"))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("rec"))
    lna.subscribe(sp)
    sp.subscribe(fusion, port="pulses")  # Spectrogram into a PulseBuffer port
    fusion.subscribe(rec)
    system.set_entry(lna)
    with pytest.raises(TypeCheckError) as exc:
        system.validate()
    msg = str(exc.value)
    assert "'sp' -> 'fusion'" in msg
    assert "Spectrogram" in msg and "PulseBuffer" in msg


def test_unknown_port_raises():
    _, system = build_system()
    lna = system.add(Amplifier("lna", gain_db=10.0))
    pd = system.add(PulseDetector("pd", threshold=0.5))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("rec"))
    lna.subscribe(pd)
    pd.subscribe(fusion, port="nope")  # not a real port
    fusion.subscribe(rec)
    system.set_entry(lna)
    with pytest.raises(TypeCheckError) as exc:
        system.validate()
    assert "unknown input port 'nope'" in str(exc.value)


def test_unfed_merge_port_raises():
    _, system = build_system()
    lna = system.add(Amplifier("lna", gain_db=10.0))
    pd = system.add(PulseDetector("pd", threshold=0.5))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("rec"))
    lna.subscribe(pd)
    pd.subscribe(fusion, port="pulses")  # spectrogram port left unfed
    fusion.subscribe(rec)
    system.set_entry(lna)
    with pytest.raises(TypeCheckError) as exc:
        system.validate()
    assert "unfed input port" in str(exc.value)
    assert "spectrogram" in str(exc.value)


def test_sink_with_subscribers_raises():
    _, system = build_system()
    lna = system.add(Amplifier("lna", gain_db=10.0))
    rec = system.add(Recorder("rec"))
    extra = system.add(Recorder("extra"))
    lna.subscribe(rec)
    rec.subscribe(extra)  # rec is a sink (produces None) but has a subscriber
    system.set_entry(lna)
    with pytest.raises(TypeCheckError) as exc:
        system.validate()
    assert "sink" in str(exc.value)


def test_entry_must_accept_signal_payload():
    _, system = build_system()
    # A pulse detector accepts SignalPayload, so use a component that does not.
    fusion = system.add(DetectionFusion("fusion"))
    system.entry = fusion  # set directly to bypass add path
    system._validated = False
    with pytest.raises(TypeCheckError) as exc:
        system.validate()
    assert "must accept SignalPayload" in str(exc.value)


def test_validation_runs_automatically_on_first_signal_rx():
    sched, system = build_system()
    lna = system.add(Amplifier("lna", gain_db=10.0))
    sp = system.add(Spectrogrammer("sp"))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("rec"))
    lna.subscribe(sp)
    sp.subscribe(fusion, port="pulses")  # bad wiring
    fusion.subscribe(rec)
    system.set_entry(lna)
    with pytest.raises(TypeCheckError):
        system.on_signal_rx(np.ones(8, dtype=np.complex64), 1e6, 1e9)


def test_aggregates_multiple_errors():
    _, system = build_system()
    lna = system.add(Amplifier("lna", gain_db=10.0))
    sp = system.add(Spectrogrammer("sp"))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("rec"))
    lna.subscribe(sp)
    sp.subscribe(fusion, port="pulses")  # type mismatch + leaves spectrogram unfed
    fusion.subscribe(rec)
    system.set_entry(lna)
    with pytest.raises(TypeCheckError) as exc:
        system.validate()
    msg = str(exc.value)
    # both the type mismatch and the unfed-port problems are reported together
    assert "produces" in msg and "unfed input port" in msg
