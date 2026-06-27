"""Step-through demo: fan-out via a splitter feeding two sinks, one event at a time.

    antenna -> LNA -> splitter -> { ADC -> recorder, spectrum analyzer }

Same topology as ``demo_fanout.py``, but this version wires with explicit
``subscribe()`` (no ``>>``), drives the scheduler one event at a time, recovers
the simulation time from the scheduler at each event, and pauses for Enter
between events.

Run interactively::

    python examples/demo_fanout_stepped.py

Or non-interactively (auto-drains)::

    printf '\\n\\n\\n\\n\\n' | python examples/demo_fanout_stepped.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem
from rfdes.components import ADC, Amplifier, Recorder, Splitter, SpectrumAnalyzer

from stepping import FireRecord, instrument, step_through


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched)
    log: list[FireRecord] = []

    lna = instrument(system.add(Amplifier("LNA", gain_db=15.0, processing_delay=1e-9)), log)
    split = instrument(system.add(Splitter("split", loss_db=3.0, processing_delay=0.5e-9)), log)
    adc = instrument(system.add(ADC("ADC", bits=10, full_scale=10.0, processing_delay=4e-9)), log)
    rec = instrument(system.add(Recorder("data")), log)
    sa = instrument(system.add(SpectrumAnalyzer("analyzer", processing_delay=0.0)), log)

    # Wire with explicit subscribe() calls (no `>>`); the splitter fans out.
    lna.subscribe(split)
    split.subscribe(adc)   # branch 1
    adc.subscribe(rec)
    split.subscribe(sa)    # branch 2
    system.set_entry(lna)

    fs = 10e6
    n = np.arange(512)
    iq = (0.05 * np.exp(2j * np.pi * 1e6 * n / fs)).astype(np.complex64)
    system.on_signal_rx(iq, sample_rate=fs, center_freq=2.4e9)

    print("Stepping through the fan-out system. Scheduler time is queried at each event.\n")
    fired = step_through(sched, log)

    print(f"\nfinished: {fired} events fired")
    print(f"final scheduler clock: {sched.now() * 1e9:.2f} ns")
    print(f"recorder received:     {len(rec.payloads)} buffer(s)")
    print(f"analyzer received:     {len(sa.payloads)} buffer(s)")


if __name__ == "__main__":
    main()
