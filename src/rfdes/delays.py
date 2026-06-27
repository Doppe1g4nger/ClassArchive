"""Factory helpers for data-dependent (and random) processing delays.

Each factory returns a callable suitable for a component's ``processing_delay``.
The callable is invoked at emit time with the component's **input** (or, for a
:class:`~rfdes.component.MergeComponent`, the matched ``{port: data}`` dict), and
returns the delay for that firing.

Example::

    from rfdes.delays import per_sample, per_pulse, jitter
    Amplifier("lna", gain_db=20, processing_delay=jitter(5e-9, 1e-9))
    PulseDetector("pd", threshold=0.5, processing_delay=per_sample(1e-9, 1e-12))
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .events import DataObject


def per_sample(base: float, per_sample: float) -> Callable[[DataObject], float]:
    """Delay = ``base + per_sample * data.num_samples``.

    For components whose input exposes ``num_samples`` (e.g. a
    :class:`~rfdes.events.SignalPayload`): models latency that grows with the
    length of the IQ buffer.
    """

    def delay(data: DataObject) -> float:
        return base + per_sample * data.num_samples

    return delay


def per_pulse(base: float, per_pulse: float) -> Callable[[DataObject], float]:
    """Delay = ``base + per_pulse * data.num_pulses``.

    For components whose input exposes ``num_pulses`` (e.g. a
    :class:`~rfdes.datatypes.PulseBuffer`): models latency that grows with the
    number of pulses to process.
    """

    def delay(data: DataObject) -> float:
        return base + per_pulse * data.num_pulses

    return delay


def jitter(
    mean: float,
    std: float,
    rng: Optional[np.random.Generator] = None,
    min_delay: float = 0.0,
) -> Callable[[object], float]:
    """Random Gaussian delay around ``mean`` (ignores the input).

    Args:
        mean: Mean delay.
        std: Standard deviation of the jitter.
        rng: A ``numpy.random.Generator`` for reproducibility; one is created if
            omitted.
        min_delay: Lower clamp so the delay never goes negative.
    """
    generator = rng if rng is not None else np.random.default_rng()

    def delay(_data: object) -> float:
        return max(min_delay, float(generator.normal(mean, std)))

    return delay
