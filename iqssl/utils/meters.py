"""Running averages and the compute accounting the fairness argument needs.

Equal epochs is not equal compute: MAE at 75% masking runs the encoder over 25%
of the tokens, two-view methods do 2x the forward passes, and BYOL/JEPA add a
teacher pass. The thesis holds epochs constant but must *report* the compute
each method actually consumed, or the comparison invites the obvious objection.
:class:`ComputeMeter` is what makes that table possible.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


class AverageMeter:
    """Running mean, count and last value."""

    __slots__ = ("count", "last", "total")

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0
        self.last = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.last = float(value)
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
        self.last = 0.0


class MeterDict:
    """A dict of :class:`AverageMeter`, keyed lazily."""

    def __init__(self) -> None:
        self._meters: dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, values: dict[str, float], n: int = 1) -> None:
        for k, v in values.items():
            self._meters[k].update(v, n)

    def averages(self) -> dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}

    def last(self) -> dict[str, float]:
        return {k: m.last for k, m in self._meters.items()}

    def reset(self) -> None:
        for m in self._meters.values():
            m.reset()

    def __getitem__(self, key: str) -> AverageMeter:
        return self._meters[key]

    def __contains__(self, key: str) -> bool:
        return key in self._meters


@dataclass
class ComputeMeter:
    """Accumulates the per-method compute figures reported alongside accuracy.

    ``encoder_forwards`` counts *token-weighted* forward passes so that MAE's
    token dropping is credited properly: a pass over 25% of the tokens costs
    0.25, not 1.0. Teacher passes are counted separately because they are
    gradient-free and roughly a third the cost of a student step.
    """

    samples_seen: int = 0
    tokens_seen: int = 0
    encoder_forwards: float = 0.0
    teacher_forwards: float = 0.0
    optimizer_steps: int = 0
    peak_memory_bytes: int = 0
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def add_forward(
        self, batch_size: int, n_tokens: int, *, teacher: bool = False, weight: float = 1.0
    ) -> None:
        """Record ``weight`` encoder passes over ``n_tokens`` tokens each.

        ``weight`` may be fractional (MAE's pass over 25% of the tokens is 0.25
        of a pass) or count several passes at once (a two-view method's step is
        weight 2). The token total scales with it so both ledgers stay
        consistent.
        """
        self.tokens_seen += round(batch_size * n_tokens * weight)
        if teacher:
            self.teacher_forwards += weight
        else:
            self.encoder_forwards += weight

    def add_step(self, batch_size: int) -> None:
        self.samples_seen += batch_size
        self.optimizer_steps += 1

    @property
    def wall_clock_s(self) -> float:
        return time.perf_counter() - self._t0

    def snapshot(self) -> dict[str, float]:
        return {
            "compute/samples_seen": float(self.samples_seen),
            "compute/tokens_seen": float(self.tokens_seen),
            "compute/encoder_forwards": self.encoder_forwards,
            "compute/teacher_forwards": self.teacher_forwards,
            "compute/optimizer_steps": float(self.optimizer_steps),
            "compute/wall_clock_s": self.wall_clock_s,
            "compute/peak_memory_gb": self.peak_memory_bytes / 1e9,
        }

    def update_peak_memory(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.peak_memory_bytes = max(self.peak_memory_bytes, torch.cuda.max_memory_allocated())


class Timer:
    """Context-manager stopwatch feeding a :class:`MeterDict`."""

    def __init__(self, meters: MeterDict, key: str) -> None:
        self._meters = meters
        self._key = key
        self._t0 = 0.0

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._meters.update({self._key: time.perf_counter() - self._t0})
