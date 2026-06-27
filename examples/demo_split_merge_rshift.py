"""Demo: split and merge wired entirely with the ``>>`` operator.

    antenna -> LNA -> split -> { PulseDetector  -> fusion["pulses"]      }
                              { Spectrogrammer -> fusion["spectrogram"] } -> recorder

The splitter fans out with repeated ``>>``; the merge's *named* input ports are
targeted with ``component[port]`` references, e.g. ``pulse_det >> fusion["pulses"]``.
This produces the same fused report as ``demo_multitype.py`` (which uses
``subscribe()``), just in the fluent ``>>`` style.

Run with::

    python examples/demo_split_merge_rshift.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem
from rfdes.components import (
    Amplifier,
    DetectionFusion,
    PulseDetector,
    Recorder,
    Spectrogrammer,
    Splitter,
)


def build_iq(fs: float) -> np.ndarray:
    n = np.arange(1024)
    tone = np.exp(2j * np.pi * 1e6 * n / fs)
    env = np.full(n.shape, 0.05)
    env[200:280] = 1.0   # burst 1
    env[600:640] = 1.0   # burst 2
    return (env * tone).astype(np.complex64)


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched)

    lna = system.add(Amplifier("LNA", gain_db=10.0, processing_delay=1e-9))
    split = system.add(Splitter("split", processing_delay=0.5e-9))
    pulses = system.add(PulseDetector("pulse-det", threshold=0.5, processing_delay=2e-9))
    spec = system.add(Spectrogrammer("spectro", nfft=64, processing_delay=3e-9))
    fusion = system.add(DetectionFusion("fusion", processing_delay=1e-9))
    rec = system.add(Recorder("report"))

    # Split (fan-out) and merge (named ports) wired purely with `>>`.
    lna >> split
    split >> pulses
    split >> spec
    pulses >> fusion["pulses"]          # target the merge's "pulses" port
    spec >> fusion["spectrogram"]       # target the merge's "spectrogram" port
    fusion >> rec
    system.set_entry(lna)

    fs = 10e6
    system.on_signal_rx(build_iq(fs), sample_rate=fs, center_freq=2.4e9)
    sched.run()

    report = rec.payloads[-1]
    print("Split + merge via >> :")
    print(f"  reports produced:  {len(rec.payloads)}")
    print(f"  pulses detected:   {report.fields['num_pulses']}")
    print(f"  peak freq:         {report.fields['peak_freq'] / 1e6:.3f} MHz")
    print(f"  spectrogram shape: {report.fields['spectrogram_shape']}")
    print(f"  final clock:       {sched.now() * 1e9:.2f} ns")


if __name__ == "__main__":
    main()
