import numpy as np
import pytest

from rfdes import HeapScheduler, PlatformState, RFSystem, SignalPayload
from rfdes.components import (
    Amplifier,
    DetectionFusion,
    JamController,
    PulseDetector,
    Spectrogrammer,
    Splitter,
    Transmitter,
)
from rfdes.datatypes import DetectionReport, PulseBuffer, Spectrogram


def report(num_pulses):
    return DetectionReport(fields={"num_pulses": num_pulses, "peak_freq": 0.0})


def rf(center_freq, n=64):
    return SignalPayload(np.ones(n, dtype=np.complex64), 1e6, center_freq)


# -- unit: the jam decision --------------------------------------------------
def test_jams_on_pulses_at_target_frequency():
    jc = JamController("jc", target_freq=2.4e9, freq_tol=1e6,
                       num_samples=32, rng=np.random.default_rng(0))
    out = jc.on_merge({"report": report(3), "rf": rf(2.4e9)})
    assert isinstance(out, SignalPayload)
    assert out.center_freq == 2.4e9
    assert out.num_samples == 32
    assert out.metadata["jam"] is True
    assert out.metadata["against_pulses"] == 3


def test_no_jam_off_frequency():
    jc = JamController("jc", target_freq=2.4e9, freq_tol=1e6)
    assert jc.on_merge({"report": report(5), "rf": rf(1.5e9)}) is None


def test_no_jam_without_pulses():
    jc = JamController("jc", target_freq=2.4e9, min_pulses=1)
    assert jc.on_merge({"report": report(0), "rf": rf(2.4e9)}) is None


# -- integration: full detect-and-jam graph ----------------------------------
def build_graph(sched, egress, jam_dwell=0.0, when_busy=None):
    system = RFSystem(sched, name="EW",
                      state=PlatformState(name="EW", position=[1, 2, 3]),
                      on_transmit=lambda p, s: egress.append((sched.now(), p, s)))
    lna = system.add(Amplifier("lna", gain_db=10.0))
    split = system.add(Splitter("split"))
    det = system.add(PulseDetector("det", threshold=0.5))
    spec = system.add(Spectrogrammer("spec", nfft=32))
    fusion = system.add(DetectionFusion("fusion"))
    jam = system.add(JamController("jam", target_freq=2.4e9, num_samples=64,
                                   rng=np.random.default_rng(0)))
    jammer = system.add(Transmitter("jammer", processing_delay=jam_dwell, when_busy=when_busy))
    lna >> split
    split >> det >> fusion["pulses"]
    split >> spec >> fusion["spectrogram"]
    split >> jam["rf"]
    fusion >> jam["report"]
    jam >> jammer
    system.set_entry(lna)
    return system, jammer


def pulsed_iq(n=256):
    k = np.arange(n)
    env = np.full(n, 0.05)
    env[50:90] = 1.0
    return (env * np.exp(2j * np.pi * 1e6 * k / 10e6)).astype(np.complex64)


def test_graph_validates_clean():
    sched = HeapScheduler()
    system, _ = build_graph(sched, [])
    system.validate()  # two merges fed, transmitter has a hook -> no raise


def test_pulsed_2p4ghz_produces_one_jam():
    sched = HeapScheduler()
    egress = []
    system, _ = build_graph(sched, egress)
    system.on_signal_rx(pulsed_iq(), sample_rate=10e6, center_freq=2.4e9)
    sched.run()
    assert len(egress) == 1
    _, payload, state = egress[0]
    assert payload.metadata["jam"] is True
    assert payload.center_freq == 2.4e9
    assert state.name == "EW"


def test_offfrequency_pulses_produce_no_jam():
    sched = HeapScheduler()
    egress = []
    system, _ = build_graph(sched, egress)
    system.on_signal_rx(pulsed_iq(), sample_rate=10e6, center_freq=1.5e9)
    sched.run()
    assert egress == []


def test_rapid_burst_drops_jams_while_busy():
    sched = HeapScheduler()
    egress = []
    system, jammer = build_graph(sched, egress, jam_dwell=5e-6, when_busy="drop")
    for i in range(5):
        sched.schedule(i * 1e-6, lambda: system.on_signal_rx(
            pulsed_iq(), sample_rate=10e6, center_freq=2.4e9))
    sched.run()
    assert len(egress) >= 1            # at least one jam got out
    assert jammer.dropped >= 1         # some were dropped while busy
    assert len(egress) + jammer.dropped == 5
