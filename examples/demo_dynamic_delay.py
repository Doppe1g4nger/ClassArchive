"""Demo: per-component delay as a dynamic function of the input (or random).

Chain::

    antenna -> LNA(jitter) -> PulseDetector(delay ~ IQ length) -> PulseRelay(delay ~ #pulses) -> recorder

Each stage's ``processing_delay`` is a callable:
  * the LNA uses ``jitter`` (random Gaussian around a mean, input-independent),
  * the PulseDetector uses ``per_sample`` (delay grows with the input IQ length),
  * a PulseBuffer->PulseBuffer relay uses ``per_pulse`` (delay grows with the
    number of pulses it must process).

The demo runs the graph for a short vs long buffer (with few vs many pulses) to
show the data-dependent delays scale, then twice on the same input to show the
random jitter differ run-to-run.

Run with::

    python examples/demo_dynamic_delay.py
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rfdes import Component, HeapScheduler, RFSystem
from rfdes.components import Amplifier, PulseDetector, Recorder
from rfdes.datatypes import PulseBuffer
from rfdes.delays import jitter, per_pulse, per_sample


class PulseRelay(Component):
    """Pass a PulseBuffer through unchanged (stand-in for a pulse processor)."""

    accepts = (PulseBuffer,)
    produces = PulseBuffer


def build_iq(n: int, num_bursts: int, fs: float = 10e6) -> np.ndarray:
    k = np.arange(n)
    tone = np.exp(2j * np.pi * 1e6 * k / fs)
    env = np.full(n, 0.05)
    width = max(n // (num_bursts * 4), 4)
    for b in range(num_bursts):
        start = (b + 1) * n // (num_bursts + 1)
        env[start:start + width] = 1.0
    return (env * tone).astype(np.complex64)


def run_once(iq: np.ndarray, seed: int) -> list[tuple[str, float]]:
    """Build and run the chain once; return [(component, arrival_time), ...]."""
    sched = HeapScheduler()
    system = RFSystem(sched)
    rng = np.random.default_rng(seed)

    lna = system.add(Amplifier("LNA", gain_db=10.0, processing_delay=jitter(5e-9, 1e-9, rng)))
    det = system.add(PulseDetector("pulse-det", threshold=0.5,
                                   processing_delay=per_sample(1e-9, 1e-11)))
    relay = system.add(PulseRelay("relay", processing_delay=per_pulse(1e-9, 5e-10)))
    rec = system.add(Recorder("sink"))

    # Tap each component's receive() to log its arrival time (= when it fires).
    log: list[tuple[str, float]] = []
    for comp in (lna, det, relay, rec):
        original = comp.receive

        def wrapped(data, port="in", _c=comp, _orig=original):
            log.append((_c.name, sched.now()))
            _orig(data, port)

        comp.receive = wrapped  # type: ignore[method-assign]

    lna >> det >> relay >> rec
    system.set_entry(lna)
    system.on_signal_rx(iq, sample_rate=10e6, center_freq=2.4e9)
    sched.run()
    return log


def show(title: str, log: list[tuple[str, float]]) -> None:
    print(title)
    # Consecutive arrival-time differences are each upstream stage's delay.
    for i in range(1, len(log)):
        stage = log[i - 1][0]
        delay = log[i][1] - log[i - 1][1]
        print(f"  {stage:<10} delay -> {delay * 1e9:6.2f} ns")
    print(f"  total latency = {log[-1][1] * 1e9:.2f} ns\n")


def main() -> None:
    short = build_iq(n=256, num_bursts=1)
    long = build_iq(n=1024, num_bursts=3)

    show("Short buffer (256 samples, 1 pulse):", run_once(short, seed=0))
    show("Long buffer (1024 samples, 3 pulses):", run_once(long, seed=0))

    print("Same input, different jitter seeds (LNA delay varies):")
    for seed in (1, 2, 3):
        log = run_once(short, seed=seed)
        lna_delay = log[1][1] - log[0][1]
        print(f"  seed {seed}: LNA delay = {lna_delay * 1e9:.3f} ns")


if __name__ == "__main__":
    main()
