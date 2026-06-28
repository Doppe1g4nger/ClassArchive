"""Capstone demo: detect pulses and jam a 2.4 GHz signal.

This combines nearly every rfdes feature into one electronic-attack platform:

  * platform 6DOF state + name on the RFSystem,
  * signalRX ingress and on_transmit egress (jam goes back to the environment),
  * multiple data types (SignalPayload, PulseBuffer, Spectrogram, DetectionReport),
  * two merges (DetectionFusion, then JamController), wired with `>>` + port refs,
  * dynamic delays (jitter on the LNA, per-sample on the detector, per-pulse on
    the jam controller) and pre-simulation type validation,
  * a Transmitter whose blocking policy (`when_busy="drop"`) models a jammer that
    is busy and cannot service every request.

Signal flow::

    signalRX -> LNA -> Splitter -+-> PulseDetector --> fusion["pulses"]
                                 +-> Spectrogrammer -> fusion["spectrogram"]
                                 +----------------------------> jam["rf"]
    DetectionFusion -> DetectionReport -> jam["report"]
    JamController (jam iff pulses AND carrier ~ 2.4 GHz) -> jammer Transmitter -> environment

Run with::

    python examples/demo_jammer.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, PlatformState, RFSystem
from rfdes.components import (
    Amplifier,
    DetectionFusion,
    JamController,
    PulseDetector,
    Spectrogrammer,
    Splitter,
    Transmitter,
)
from rfdes.delays import jitter, per_sample

FS = 10e6
JAM_DWELL = 3e-6  # how long the jammer is busy transmitting one jam


def pulsed(center_freq: float, n: int = 1024, num_bursts: int = 3) -> dict:
    """A pulsed signal (bursts above the detector threshold)."""
    k = np.arange(n)
    tone = np.exp(2j * np.pi * 1e6 * k / FS)
    env = np.full(n, 0.05)
    width = max(n // (num_bursts * 5), 4)
    for b in range(num_bursts):
        start = (b + 1) * n // (num_bursts + 1)
        env[start:start + width] = 1.0
    return dict(iq=(env * tone).astype(np.complex64), sample_rate=FS, center_freq=center_freq)


def noise_only(center_freq: float, n: int = 1024) -> dict:
    """Low-level noise: stays below the detector threshold even after LNA gain."""
    rng = np.random.default_rng(0)
    iq = (0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    return dict(iq=iq, sample_rate=FS, center_freq=center_freq)


def build_system(sched: HeapScheduler, egress: list) -> tuple[RFSystem, Transmitter]:
    def on_transmit(payload, state):
        egress.append((sched.now(), payload, state))

    state = PlatformState(name="EW-Platform",
                          position=[5000.0, 1200.0, 8000.0],
                          velocity=[200.0, 0.0, 0.0],
                          orientation=[1.0, 0.0, 0.0, 0.0])
    system = RFSystem(sched, name="EW-Platform", state=state, on_transmit=on_transmit)

    rng = np.random.default_rng(42)
    lna = system.add(Amplifier("LNA", gain_db=20.0, processing_delay=jitter(2e-9, 5e-10, rng)))
    split = system.add(Splitter("split", processing_delay=1e-9))
    det = system.add(PulseDetector("pulse-det", threshold=0.5,
                                   processing_delay=per_sample(1e-9, 1e-12)))
    spec = system.add(Spectrogrammer("spectro", nfft=64, processing_delay=2e-9))
    fusion = system.add(DetectionFusion("fusion", processing_delay=1e-9))
    jam = system.add(JamController(
        "jam-ctrl", target_freq=2.4e9, freq_tol=1e6, min_pulses=1,
        num_samples=256, rng=np.random.default_rng(7),
        processing_delay=lambda inp: 1e-9 + 5e-10 * inp["report"].fields["num_pulses"],
    ))
    jammer = system.add(Transmitter("jammer", processing_delay=JAM_DWELL, when_busy="drop"))

    # Wire entirely with `>>` (port refs target the merges' named inputs).
    lna >> split
    split >> det >> fusion["pulses"]
    split >> spec >> fusion["spectrogram"]
    split >> jam["rf"]
    fusion >> jam["report"]
    jam >> jammer
    system.set_entry(lna)
    return system, jammer


def scenario(title: str, system: RFSystem, sched: HeapScheduler, egress: list, sig: dict) -> None:
    before = len(egress)
    system.on_signal_rx(**sig)
    sched.run()
    new = egress[before:]
    if new:
        t, payload, state = new[-1]
        print(f"{title}\n  -> JAMMED at t={t*1e9:.1f} ns: {payload.num_samples}-sample "
              f"barrage @ {payload.center_freq/1e9:.3f} GHz vs "
              f"{payload.metadata['against_pulses']} pulses; "
              f"TX from {state.name!r} at {state.position.tolist()}\n")
    else:
        print(f"{title}\n  -> no jam\n")


def main() -> None:
    sched = HeapScheduler()
    egress: list = []
    system, jammer = build_system(sched, egress)

    print("EW platform: detect pulses, jam only pulsed signals at 2.4 GHz.\n")
    scenario("1) Pulsed signal @ 2.4 GHz", system, sched, egress, pulsed(2.4e9))
    scenario("2) Pulsed signal @ 1.5 GHz (off target)", system, sched, egress, pulsed(1.5e9))
    scenario("3) Noise only @ 2.4 GHz (no pulses)", system, sched, egress, noise_only(2.4e9))

    # 4) Rapid burst of pulsed 2.4 GHz signals faster than the jammer's dwell.
    print("4) Rapid burst of pulsed @ 2.4 GHz (jammer can't keep up)")
    before = len(egress)
    base = sched.now()
    for i in range(5):
        system.on_signal_rx(**pulsed(2.4e9), at=base + i * 1e-6)
    sched.run()
    emitted = len(egress) - before
    print(f"  -> {emitted} jam(s) emitted, {jammer.dropped} dropped while busy "
          f"(dwell={JAM_DWELL*1e9:.0f} ns)\n")
    print(f"total transmissions to environment: {len(egress)}")


if __name__ == "__main__":
    main()
