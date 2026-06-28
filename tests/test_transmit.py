import numpy as np
import pytest

from rfdes import Component, HeapScheduler, PlatformState, RFSystem, SignalPayload, TypeCheckError
from rfdes.components import Repeater, ToneTransmitter, Transmitter


# -- PlatformState ------------------------------------------------------------
def test_platform_state_coerces_inputs():
    s = PlatformState(name="p", position=[1, 2, 3], velocity=(4, 5, 6))
    assert isinstance(s.position, np.ndarray) and s.position.tolist() == [1, 2, 3]
    assert isinstance(s.velocity, np.ndarray) and s.velocity.tolist() == [4, 5, 6]
    assert s.orientation.tolist() == [1.0, 0.0, 0.0, 0.0]  # identity default


def test_platform_state_rejects_bad_shapes():
    with pytest.raises(ValueError):
        PlatformState(position=[1, 2])
    with pytest.raises(ValueError):
        PlatformState(orientation=[1, 0, 0])


def test_snapshot_is_independent():
    s = PlatformState(name="p", position=[0, 0, 0])
    snap = s.snapshot()
    s.position[0] = 99
    assert snap.position[0] == 0.0


# -- RFSystem state + back-reference -----------------------------------------
def test_system_carries_name_and_state():
    sched = HeapScheduler()
    state = PlatformState(name="A", position=[1, 2, 3])
    system = RFSystem(sched, name="A", state=state)
    assert system.name == "A"
    assert system.state.position.tolist() == [1, 2, 3]


def test_add_sets_back_reference():
    sched = HeapScheduler()
    system = RFSystem(sched, name="A")
    c = system.add(Component("c"))
    assert c.system is system


# -- transmit egress ----------------------------------------------------------
def capture():
    log = []
    return log, (lambda payload, state: log.append((payload, state)))


def test_repeater_relay_transmits_with_state_and_delay():
    sched = HeapScheduler()
    log, hook = capture()
    times = []
    state = PlatformState(name="A", position=[10, 0, 0])
    system = RFSystem(sched, name="A", state=state,
                      on_transmit=lambda p, s: (times.append(sched.now()), log.append((p, s))))
    rep = system.add(Repeater("rep", gain_db=20.0, processing_delay=5.0))
    system.set_entry(rep)

    iq = np.ones(8, dtype=np.complex64)
    system.on_signal_rx(iq, 1e6, 1e9)
    sched.run()

    assert len(log) == 1
    payload, snap = log[0]
    assert np.allclose(np.abs(payload.iq), 10.0)          # +20 dB gain applied
    assert snap.name == "A" and snap.position.tolist() == [10, 0, 0]
    assert times == [5.0]                                 # egress at processing_delay


def test_tone_transmitter_source_mode():
    sched = HeapScheduler()
    log, hook = capture()
    system = RFSystem(sched, name="B", on_transmit=hook)
    beacon = system.add(ToneTransmitter("beacon", freq=1e6, sample_rate=10e6,
                                        num_samples=128, center_freq=2.4e9,
                                        processing_delay=2.0))
    beacon.fire()
    sched.run()
    assert len(log) == 1
    payload, snap = log[0]
    assert payload.num_samples == 128
    assert payload.center_freq == 2.4e9
    assert payload.metadata["tx"] == "beacon"


def test_transmit_without_hook_raises():
    sched = HeapScheduler()
    system = RFSystem(sched, name="A")  # no on_transmit
    with pytest.raises(RuntimeError):
        system.transmit(SignalPayload(np.ones(4, dtype=np.complex64), 1e6, 1e9))


def test_validate_flags_transmitter_without_hook():
    sched = HeapScheduler()
    system = RFSystem(sched, name="A")  # no on_transmit
    rep = system.add(Repeater("rep"))
    system.set_entry(rep)
    with pytest.raises(TypeCheckError) as exc:
        system.validate()
    assert "on_transmit" in str(exc.value)


def test_snapshot_reflects_state_at_transmit_time():
    sched = HeapScheduler()
    log, hook = capture()
    system = RFSystem(sched, name="A",
                      state=PlatformState(name="A", position=[0, 0, 0]),
                      on_transmit=hook)
    beacon = system.add(ToneTransmitter("beacon", freq=1e6, sample_rate=10e6,
                                        num_samples=16))
    system.state.position = np.array([5.0, 6.0, 7.0])  # platform moved before TX
    beacon.fire()
    sched.run()
    _, snap = log[0]
    assert snap.position.tolist() == [5.0, 6.0, 7.0]
    # later motion does not change the captured snapshot
    system.state.position[0] = 999
    assert snap.position[0] == 5.0
