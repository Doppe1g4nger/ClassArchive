import numpy as np
import pytest

from rfdes import (
    Component,
    ControllableComponent,
    ControlMessage,
    HeapScheduler,
    RFSystem,
    SignalPayload,
)
from rfdes.component import CONTROL_PORT
from rfdes.components import PulseDetector, Recorder, ScanScheduler, TunableBandpassFilter
from rfdes.datatypes import PulseBuffer


FS = 10e6


def tone(freq, n=1024, amp=1.0):
    k = np.arange(n)
    return (amp * np.exp(2j * np.pi * freq * k / FS)).astype(np.complex64)


# -- ControllableComponent base ----------------------------------------------
class RecordingControllable(ControllableComponent):
    def __init__(self, name):
        super().__init__(name)
        self.controls = []

    def on_control(self, msg):
        self.controls.append(msg)


def test_control_port_in_input_ports():
    c = RecordingControllable("c")
    ports = c.input_ports()
    assert CONTROL_PORT in ports
    assert ControlMessage in ports[CONTROL_PORT]
    assert "in" in ports  # signal port still present


def test_control_message_updates_state_without_emitting():
    sched = HeapScheduler()
    system = RFSystem(sched)
    c = system.add(RecordingControllable("c"))
    rec = system.add(Recorder("rec"))
    c.subscribe(rec)

    c.receive(ControlMessage(params={"x": 1}), port=CONTROL_PORT)
    sched.run()
    assert len(c.controls) == 1          # on_control ran
    assert rec.payloads == []            # control produced no downstream output


def test_default_port_still_emits():
    sched = HeapScheduler()
    system = RFSystem(sched)
    c = system.add(RecordingControllable("c"))
    rec = system.add(Recorder("rec"))
    c.subscribe(rec)

    c.receive(SignalPayload(tone(1e6), FS, 2.4e9))  # default port
    sched.run()
    assert len(rec.payloads) == 1
    assert c.controls == []


# -- TunableBandpassFilter ----------------------------------------------------
def test_tunable_filter_retunes_via_control():
    filt = TunableBandpassFilter("f", bandwidth=2e6, passband_center=0.0)
    # n=1000 places the 2 MHz tone on an exact FFT bin (no leakage into passband).
    sig = SignalPayload(tone(2e6, n=1000), FS, 2.4e9)

    blocked = filt.on_signal(sig)        # 2 MHz tone, passband at 0 -> rejected
    assert np.max(np.abs(blocked.iq)) < 0.1

    filt.on_control(ControlMessage(params={"passband_center": 2e6}))
    assert filt.passband_center == 2e6
    passed = filt.on_signal(sig)         # now in band -> passes
    assert np.max(np.abs(passed.iq)) > 0.5


# -- ScanScheduler ------------------------------------------------------------
def test_scan_advances_when_no_pulses():
    scan = ScanScheduler("s", bands=[-2e6, 0.0, 2e6], bandwidth=2e6, start_index=0)
    empty = PulseBuffer(pulses=np.zeros((0, 3)), sample_rate=FS)
    msg = scan.on_signal(empty)
    assert isinstance(msg, ControlMessage)
    assert scan.index == 1
    assert msg.params["passband_center"] == 0.0
    assert scan.locked is False


def test_scan_dwells_on_detection():
    scan = ScanScheduler("s", bands=[-2e6, 0.0, 2e6], bandwidth=2e6, start_index=2)
    pulses = PulseBuffer(pulses=np.ones((2, 3)), sample_rate=FS)
    assert scan.on_signal(pulses) is None   # dwell, no control
    assert scan.index == 2
    assert scan.locked is True


# -- closed-loop integration --------------------------------------------------
def build_loop():
    sched = HeapScheduler()
    system = RFSystem(sched, name="scan-rx")
    bands = [-4e6, -2e6, 0.0, 2e6, 4e6]
    filt = system.add(TunableBandpassFilter("filt", bandwidth=2e6, passband_center=bands[0]))
    det = system.add(PulseDetector("det", threshold=0.5))
    rec = system.add(Recorder("rec"))
    scan = system.add(ScanScheduler("scan", bands=bands, bandwidth=2e6, start_index=0))
    filt >> det
    det >> rec
    det >> scan
    scan >> filt["control"]
    system.set_entry(filt)
    return sched, system, filt, rec, scan


def pulsed():
    carrier = tone(2e6)
    env = np.full(1024, 0.05)
    env[200:240] = 1.0
    env[600:640] = 1.0
    return (env * carrier).astype(np.complex64)


def test_feedback_graph_validates_clean():
    _, system, _, _, _ = build_loop()
    system.validate()  # the feedback cycle (control edge) must not raise


def test_closed_loop_acquires_and_locks():
    sched, system, filt, rec, scan = build_loop()
    sig = pulsed()
    seen = []
    for _ in range(6):
        band = filt.passband_center
        system.on_signal_rx(sig, sample_rate=FS, center_freq=2.4e9)
        sched.run()
        seen.append((band, rec.payloads[-1].num_pulses))

    # locks onto the +2 MHz band (where the tone is) and dwells there
    assert scan.locked is True
    assert filt.passband_center == 2e6
    # pulses are only ever detected while tuned to the signal's band
    assert any(p > 0 for _, p in seen)
    assert all(band == 2e6 for band, p in seen if p > 0)
