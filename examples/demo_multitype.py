"""Demo: multiple data types, type validation, and a merge component.

    antenna -> LNA -> splitter -> { PulseDetector  -> pulses     }
                                  { Spectrogrammer -> spectrogram } -> DetectionFusion -> recorder

The splitter fans IQ to two branches that produce *different* data types
(a PulseBuffer and a Spectrogram). DetectionFusion is a merge component: it fires
only once it has received both products, then emits a fused DetectionReport.

The second half shows the pre-simulation type check rejecting a mis-wired graph.

Run with::

    python examples/demo_multitype.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem, TypeCheckError
from rfdes.components import (
    Amplifier,
    DetectionFusion,
    PulseDetector,
    Recorder,
    Spectrogrammer,
    Splitter,
)


def build_iq(fs: float) -> np.ndarray:
    """A buffer with two short bursts so the pulse detector finds pulses."""
    n = np.arange(1024)
    tone = np.exp(2j * np.pi * 1e6 * n / fs)
    env = np.full(n.shape, 0.05)
    env[200:280] = 1.0   # burst 1
    env[600:640] = 1.0   # burst 2
    return (env * tone).astype(np.complex64)


def run_fusion_demo() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched)

    lna = system.add(Amplifier("LNA", gain_db=10.0, processing_delay=1e-9))
    split = system.add(Splitter("split", processing_delay=0.5e-9))
    pulses = system.add(PulseDetector("pulse-det", threshold=0.5, processing_delay=2e-9))
    spec = system.add(Spectrogrammer("spectro", nfft=64, processing_delay=3e-9))
    fusion = system.add(DetectionFusion("fusion", processing_delay=1e-9))
    rec = system.add(Recorder("report"))

    # Explicit subscribe() wiring; the merge ports are named.
    lna.subscribe(split)
    split.subscribe(pulses)
    split.subscribe(spec)
    pulses.subscribe(fusion, port="pulses")
    spec.subscribe(fusion, port="spectrogram")
    fusion.subscribe(rec)
    system.set_entry(lna)

    fs = 10e6
    system.on_signal_rx(build_iq(fs), sample_rate=fs, center_freq=2.4e9)
    sched.run()

    report = rec.payloads[-1]
    print("Fusion demo (multiple data types + merge):")
    print(f"  reports produced:  {len(rec.payloads)}")
    print(f"  pulses detected:   {report.fields['num_pulses']}")
    print(f"  peak freq:         {report.fields['peak_freq'] / 1e6:.3f} MHz")
    print(f"  spectrogram shape: {report.fields['spectrogram_shape']}")
    print(f"  final clock:       {sched.now() * 1e9:.2f} ns")


def run_validation_demo() -> None:
    """Wire a spectrogram into the pulse port (wrong type) and let validate() catch it."""
    sched = HeapScheduler()
    system = RFSystem(sched)

    lna = system.add(Amplifier("LNA", gain_db=10.0))
    spec = system.add(Spectrogrammer("spectro"))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("report"))

    lna.subscribe(spec)
    spec.subscribe(fusion, port="pulses")   # WRONG: spectrogram into pulses port
    fusion.subscribe(rec)
    system.set_entry(lna)

    print("\nValidation demo (deliberately mis-wired):")
    try:
        system.validate()
    except TypeCheckError as exc:
        print("  caught TypeCheckError:")
        for line in str(exc).splitlines():
            print(f"    {line}")


if __name__ == "__main__":
    run_fusion_demo()
    run_validation_demo()
