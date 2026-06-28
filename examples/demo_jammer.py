"""Capstone demo: scan, detect pulses, and jam a 2.4 GHz signal.

This combines essentially every rfdes feature into one electronic-attack platform:

  * platform 6DOF state + name on the RFSystem,
  * signalRX ingress (incl. scheduling at an absolute time via ``at=``) and
    on_transmit egress (the jam goes back to the environment),
  * multi-channel ``(channels, N)`` IQ flowing through the chain,
  * multiple data types (SignalPayload, PulseBuffer, Spectrogram, ControlMessage,
    DetectionReport), wired with ``>>`` + port refs,
  * two merges (DetectionFusion, then JamController),
  * **closed-loop feedback**: a ScanScheduler retunes a TunableBandpassFilter
    front-end based on whether the detector saw pulses (scan-acquire-then-jam),
  * dynamic delays (jitter on the LNA, per-sample on the detector, per-pulse on
    the jam controller) and pre-simulation type validation,
  * a Transmitter whose blocking policy (``when_busy="drop"``) models a jammer
    that is busy and cannot service every request,
  * event-queue introspection with ``scheduler.print_queue()``.

Signal flow::

    signalRX -> LNA -> TunableBandpassFilter -> Splitter -+-> PulseDetector -+-> fusion["pulses"]
                              ^                            |                  +-> ScanScheduler --,
                              |                            +-> Spectrogrammer -> fusion["spectrogram"]
                              |                            +----------------------------> jam["rf"]
                              |   DetectionFusion -> DetectionReport -> jam["report"]
                              |   JamController (jam iff pulses AND carrier ~ 2.4 GHz) -> jammer -> environment
                              '----------------- filter["control"] <----------------------'

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
    Recorder,
    ScanScheduler,
    Spectrogrammer,
    Splitter,
    Transmitter,
    TunableBandpassFilter,
)
from rfdes.delays import jitter, per_sample

FS = 10e6
SIGNAL_TONE = 2e6          # baseband offset of the threat signal
BANDS = [-4e6, -2e6, 0.0, 2e6, 4e6]
BANDWIDTH = 2e6
JAM_DWELL = 3e-6           # how long the jammer is busy transmitting one jam


def pulsed(center_freq: float, n: int = 1024, num_bursts: int = 3) -> dict:
    """A 2-channel pulsed signal: bursts modulated on a +2 MHz baseband tone."""
    k = np.arange(n)
    tone = np.exp(2j * np.pi * SIGNAL_TONE * k / FS)
    env = np.full(n, 0.02)
    width = max(n // (num_bursts * 5), 4)
    for b in range(num_bursts):
        start = (b + 1) * n // (num_bursts + 1)
        env[start:start + width] = 1.0
    ch0 = env * tone
    ch1 = 0.5 * env * tone                       # second antenna, weaker
    iq = np.stack([ch0, ch1]).astype(np.complex64)  # (channels, N)
    return dict(iq=iq, sample_rate=FS, center_freq=center_freq)


def build_system(sched: HeapScheduler, egress: list):
    def on_transmit(payload, state):
        egress.append((sched.now(), payload, state))

    state = PlatformState(name="EW-Platform",
                          position=[5000.0, 1200.0, 8000.0],
                          velocity=[200.0, 0.0, 0.0],
                          orientation=[1.0, 0.0, 0.0, 0.0])
    system = RFSystem(sched, name="EW-Platform", state=state, on_transmit=on_transmit)

    rng = np.random.default_rng(42)
    lna = system.add(Amplifier("LNA", gain_db=20.0, processing_delay=jitter(2e-9, 5e-10, rng)))
    filt = system.add(TunableBandpassFilter("scan-filter", bandwidth=BANDWIDTH,
                                            passband_center=BANDS[0], processing_delay=1e-9))
    split = system.add(Splitter("split", processing_delay=1e-9))
    # Threshold set above the out-of-band leakage floor (sharp bursts are
    # broadband), so the detector only fires when the filter is tuned to the tone.
    det = system.add(PulseDetector("pulse-det", threshold=5.0,
                                   processing_delay=per_sample(1e-9, 1e-12)))
    spec = system.add(Spectrogrammer("spectro", nfft=64, processing_delay=2e-9))
    fusion = system.add(DetectionFusion("fusion", processing_delay=1e-9))
    scan = system.add(ScanScheduler("scan-sched", bands=BANDS, bandwidth=BANDWIDTH,
                                    start_index=0, processing_delay=1e-9))
    pulse_rec = system.add(Recorder("pulse-tap"))
    jam = system.add(JamController(
        "jam-ctrl", target_freq=2.4e9, freq_tol=1e6, min_pulses=1,
        num_samples=256, rng=np.random.default_rng(7),
        processing_delay=lambda inp: 1e-9 + 5e-10 * inp["report"].fields["num_pulses"],
    ))
    jammer = system.add(Transmitter("jammer", processing_delay=JAM_DWELL, when_busy="drop"))

    # Wire entirely with `>>` (port refs target named merge / control inputs).
    lna >> filt >> split
    split >> det
    split >> spec >> fusion["spectrogram"]
    split >> jam["rf"]
    det >> fusion["pulses"]
    det >> pulse_rec
    det >> scan
    scan >> filt["control"]          # closed-loop feedback retunes the front-end
    fusion >> jam["report"]
    jam >> jammer
    system.set_entry(lna)
    return system, filt, scan, pulse_rec, jammer


def main() -> None:
    sched = HeapScheduler()
    egress: list = []
    system, filt, scan, pulse_rec, jammer = build_system(sched, egress)

    print("EW platform: scan for a pulsed 2.4 GHz threat, then jam it.")
    print("(2-channel IQ; front-end filter retuned by closed-loop feedback)\n")

    # 1) Scan-acquire-then-jam: feed look-through buffers of the 2.4 GHz threat.
    print("1) Scanning for the threat (pulsed @ 2.4 GHz):")
    for i in range(6):
        band = filt.passband_center
        before = len(egress)
        system.on_signal_rx(**pulsed(2.4e9))
        sched.run()
        pulses = pulse_rec.payloads[-1].num_pulses
        jammed = len(egress) > before
        state = "lock " if scan.locked else "scan "
        action = "JAMMED" if jammed else "------"
        print(f"   look {i}: filter {band/1e6:+.0f} MHz  {state} {pulses} pulse(s)  {action}")
    print()

    # 2) Off-target carrier: detector still sees pulses (filter is locked), but
    #    the JamController gates on carrier frequency, so no jam.
    before = len(egress)
    system.on_signal_rx(**pulsed(1.5e9))
    sched.run()
    print(f"2) Pulsed @ 1.5 GHz (off target): "
          f"{'JAMMED' if len(egress) > before else 'no jam (carrier gated)'}\n")

    # 3) Rapid burst once locked -> the jammer can't keep up; inspect the queue.
    print("3) Rapid burst of pulsed @ 2.4 GHz (faster than the jammer's dwell):")
    before = len(egress)
    base = sched.now()
    for i in range(5):
        system.on_signal_rx(**pulsed(2.4e9), at=base + i * 1e-6)
    print("   queued events right after scheduling the burst:")
    for line in sched.format_queue().splitlines():
        print(f"     {line}")
    sched.run()
    emitted = len(egress) - before
    print(f"   -> {emitted} jam(s) emitted, {jammer.dropped} dropped while busy "
          f"(dwell={JAM_DWELL*1e9:.0f} ns)\n")

    print(f"final band = {filt.passband_center/1e6:+.0f} MHz, locked = {scan.locked}")
    print(f"total transmissions to environment: {len(egress)}")


if __name__ == "__main__":
    main()
