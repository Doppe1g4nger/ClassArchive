"""Step-through demo: a linear RF receive chain, one event at a time.

    antenna -> LNA -> mixer -> ADC -> recorder

Same topology as ``demo_chain.py``, but this version:
  * wires components with explicit ``subscribe()`` (no ``>>`` sugar),
  * drives the scheduler one event at a time with ``step()``,
  * recovers the current simulation time from the scheduler at each event,
  * pauses for Enter between events.

Run interactively::

    python examples/demo_chain_stepped.py

Or non-interactively (auto-drains)::

    printf '\\n\\n\\n\\n' | python examples/demo_chain_stepped.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem
from rfdes.components import ADC, Amplifier, Mixer, Recorder

from stepping import FireRecord, instrument, step_through


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched)
    log: list[FireRecord] = []

    # Per-component processing delays (seconds) model each stage's latency.
    lna = instrument(system.add(Amplifier("LNA", gain_db=20.0, processing_delay=1e-9)), log)
    mixer = instrument(system.add(Mixer("mixer", lo_freq=1.0e9, processing_delay=2e-9)), log)
    adc = instrument(system.add(ADC("ADC", bits=12, full_scale=20.0, processing_delay=5e-9)), log)
    rec = instrument(system.add(Recorder("baseband")), log)

    # Wire the chain with explicit subscribe() calls (no `>>`).
    lna.subscribe(mixer)
    mixer.subscribe(adc)
    adc.subscribe(rec)
    system.set_entry(lna)

    # Build an input IQ buffer: a tone at +2 MHz, sampled at 10 Msps.
    fs = 10e6
    n = np.arange(1024)
    iq = (0.1 * np.exp(2j * np.pi * 2e6 * n / fs)).astype(np.complex64)

    # The "RF environment" raises signalRX with a 1.5 GHz carrier.
    system.on_signal_rx(iq, sample_rate=fs, center_freq=1.5e9, t=100e-9)

    print("Stepping through the receive chain. Scheduler time is queried at each event.\n")
    fired = step_through(sched, log)

    out = rec.payloads[-1]
    print(f"\nfinished: {fired} events fired")
    print(f"final scheduler clock: {sched.now() * 1e9:.1f} ns")
    print(f"output center freq:    {out.center_freq / 1e9:.3f} GHz")
    print(f"cumulative gain:       {out.metadata['gain_db']:.1f} dB")


if __name__ == "__main__":
    main()
