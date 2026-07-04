"""Demo: measure each component's core-execution wall-clock time.

Every component times only its *core execution* -- the user transform
(``on_signal`` / ``on_merge`` / ``on_control``). The framework's data prep
(payload copies, ``start_time`` stamping) and message passing (scheduling,
fan-out) are excluded, so the numbers reflect genuine compute cost.

Here a cheap vectorized amplifier is compared with a Python-loop pulse detector
and FFT-based spectrogrammer over a large buffer.

Run with::

    python examples/demo_timing.py
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

FS = 10e6


def signal(n: int = 200_000) -> np.ndarray:
    k = np.arange(n)
    env = np.full(n, 0.05)
    for b in range(20):
        s = (b + 1) * n // 21
        env[s:s + 300] = 1.0
    return (env * np.exp(2j * np.pi * 2e6 * k / FS)).astype(np.complex64)


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched, name="timing-demo")

    lna = system.add(Amplifier("LNA", gain_db=10.0))
    split = system.add(Splitter("split"))
    det = system.add(PulseDetector("pulse-det", threshold=0.5))     # Python loop
    spec = system.add(Spectrogrammer("spectro", nfft=1024))         # many FFTs
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("report"))

    lna >> split
    split >> det >> fusion["pulses"]
    split >> spec >> fusion["spectrogram"]
    fusion >> rec
    system.set_entry(lna)

    # Run several buffers so the averages are meaningful.
    for _ in range(5):
        system.on_signal_rx(signal(), sample_rate=FS, center_freq=2.4e9)
        sched.run()

    print("Per-component core-execution time (transform only, over 5 buffers):\n")
    system.print_timing()

    report = system.timing_report()
    hottest = report[0]
    total = sum(r.total for r in report)
    print(f"\nhottest: {hottest.name} "
          f"({hottest.total * 1e3:.1f} ms, {100 * hottest.total / total:.0f}% of core time)")


if __name__ == "__main__":
    main()
