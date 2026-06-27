"""Demo: fan-out via a splitter feeding two independent sinks.

    antenna -> LNA -> splitter -> { ADC -> recorder, spectrum analyzer }

Run with::

    python examples/demo_fanout.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem
from rfdes.components import ADC, Amplifier, Recorder, Splitter, SpectrumAnalyzer


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched)

    lna = system.add(Amplifier("LNA", gain_db=15.0, processing_delay=1e-9))
    split = system.add(Splitter("split", loss_db=3.0, processing_delay=0.5e-9))
    adc = system.add(ADC("ADC", bits=10, full_scale=10.0, processing_delay=4e-9))
    rec = system.add(Recorder("data"))
    sa = system.add(SpectrumAnalyzer("analyzer", processing_delay=0.0))

    lna >> split
    split >> adc >> rec   # branch 1
    split >> sa           # branch 2
    system.set_entry(lna)

    fs = 10e6
    n = np.arange(512)
    iq = (0.05 * np.exp(2j * np.pi * 1e6 * n / fs)).astype(np.complex64)
    system.on_signal_rx(iq, sample_rate=fs, center_freq=2.4e9)
    sched.run()

    print(f"recorder received:  {len(rec.payloads)} buffer(s)")
    print(f"analyzer received:  {len(sa.payloads)} buffer(s)")
    spec = sa.last_spectrum()
    peak_bin = int(np.argmax(spec))
    print(f"analyzer peak bin:  {peak_bin} / {spec.size}")
    print(f"final clock:        {sched.now() * 1e9:.2f} ns")


if __name__ == "__main__":
    main()
