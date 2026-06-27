"""Demo: a linear RF receive chain driven by the reference scheduler.

    antenna -> LNA -> mixer -> ADC -> recorder

Run with::

    python examples/demo_chain.py

This stands in for the external RF-environment simulator by using
``HeapScheduler`` and calling ``on_signal_rx`` directly. In production the
external simulator would call ``on_signal_rx`` on its ``signalRX`` event and own
the queue itself.
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, RFSystem
from rfdes.components import ADC, Amplifier, Mixer, Recorder


def main() -> None:
    sched = HeapScheduler()
    system = RFSystem(sched)

    # Per-component processing delays (seconds) model each stage's latency.
    lna = system.add(Amplifier("LNA", gain_db=20.0, processing_delay=1e-9))
    mixer = system.add(Mixer("mixer", lo_freq=1.0e9, processing_delay=2e-9))
    adc = system.add(ADC("ADC", bits=12, full_scale=20.0, processing_delay=5e-9))
    rec = system.add(Recorder("baseband"))

    # Wire the chain. `>>` returns the downstream component for chaining.
    lna >> mixer >> adc >> rec
    system.set_entry(lna)

    # Build an input IQ buffer: a tone at +2 MHz, sampled at 10 Msps.
    fs = 10e6
    n = np.arange(1024)
    iq = (0.1 * np.exp(2j * np.pi * 2e6 * n / fs)).astype(np.complex64)

    # The "RF environment" raises signalRX at t=100 ns with a 1.5 GHz carrier.
    system.on_signal_rx(iq, sample_rate=fs, center_freq=1.5e9, t=100e-9)
    sched.run()

    out = rec.payloads[-1]
    total_delay = lna.processing_delay + mixer.processing_delay + adc.processing_delay
    print(f"events delivered: {len(rec.payloads)}")
    print(f"input center freq:   1.500 GHz")
    print(f"output center freq:  {out.center_freq / 1e9:.3f} GHz (after LO=1.0 GHz)")
    print(f"cumulative gain:     {out.metadata['gain_db']:.1f} dB")
    print(f"ADC bits:            {out.metadata['adc_bits']}")
    print(f"input start_time:    100.0 ns")
    print(f"output start_time:   {out.start_time * 1e9:.1f} ns")
    print(f"chain latency:       {total_delay * 1e9:.1f} ns")
    print(f"scheduler clock:     {sched.now() * 1e9:.1f} ns")


if __name__ == "__main__":
    main()
