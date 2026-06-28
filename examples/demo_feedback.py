"""Demo: closed-loop feedback — a scan scheduler retunes a front-end filter.

A tunable band-pass filter scans through frequency bands. After each buffer, the
pulse detector's result is fed back to a ScanScheduler, which either dwells (if
pulses were seen) or retunes the filter to the next band (if not). The loop
acquires and locks onto the band containing the signal.

    signalRX -> TunableBandpassFilter -> PulseDetector -+-> Recorder
                                                        +-> ScanScheduler -> filter["control"]

Run with::

    python examples/demo_feedback.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem
from rfdes.components import PulseDetector, Recorder, ScanScheduler, TunableBandpassFilter

FS = 10e6
SIGNAL_TONE = 2e6  # baseband offset of the signal we want to find


def pulsed_signal() -> np.ndarray:
    """Bursts modulated on a +2 MHz tone (so only the +2 MHz band passes them)."""
    k = np.arange(1024)
    carrier = np.exp(2j * np.pi * SIGNAL_TONE * k / FS)
    env = np.full(1024, 0.05)
    env[200:240] = 1.0
    env[600:640] = 1.0
    return (env * carrier).astype(np.complex64)


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched, name="Scan-Rx")

    bands = [-4e6, -2e6, 0.0, 2e6, 4e6]
    bw = 2e6
    filt = system.add(TunableBandpassFilter("filter", bandwidth=bw,
                                            passband_center=bands[0], processing_delay=1e-9))
    det = system.add(PulseDetector("detector", threshold=0.5, processing_delay=1e-9))
    rec = system.add(Recorder("recorder"))
    scan = system.add(ScanScheduler("scan-sched", bands=bands, bandwidth=bw,
                                    start_index=0, processing_delay=1e-9))

    # Closed loop wired with `>>` (feedback edge targets the filter's control port).
    filt >> det
    det >> rec
    det >> scan
    scan >> filt["control"]
    system.set_entry(filt)

    sig = pulsed_signal()
    print(f"Scanning bands {[b/1e6 for b in bands]} MHz for a signal at "
          f"{SIGNAL_TONE/1e6:.0f} MHz (bandwidth {bw/1e6:.0f} MHz):\n")

    for i in range(6):
        band_used = filt.passband_center           # band active for this buffer
        system.on_signal_rx(sig, sample_rate=FS, center_freq=2.4e9)
        sched.run()                                # feedback applies during the drain
        pulses = rec.payloads[-1].num_pulses
        action = "DWELL (locked)" if scan.locked else f"advance -> {filt.passband_center/1e6:+.0f} MHz"
        print(f"  buffer {i}: band {band_used/1e6:+.0f} MHz -> {pulses} pulse(s); {action}")

    print(f"\nlocked = {scan.locked}, final band = {filt.passband_center/1e6:+.0f} MHz")


if __name__ == "__main__":
    main()
