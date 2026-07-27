"""Exponential-moving-average teacher for BYOL, data2vec and the JEPAs.

Concentrates two bugs that are individually silent and jointly ruinous.

**EMA the parameters; *copy* the buffers.** BatchNorm running statistics are
buffers, and exponentially averaging them produces a teacher whose normalization
lags its own weights. Training still converges, just to a worse solution — which
looks exactly like "this method underperforms on IQ data."

**The teacher must never receive gradients.** It is not enough to wrap the
forward in ``no_grad``: the teacher's parameters must have ``requires_grad =
False``, or an optimizer constructed over ``model.parameters()`` will pick them
up and quietly train the target alongside the student.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
from torch import nn

ScheduleKind = Literal["constant", "linear", "cosine"]


class EMATeacher(nn.Module):
    """A frozen, slowly-updated copy of a student module."""

    def __init__(
        self,
        student: nn.Module,
        *,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
        schedule: ScheduleKind = "cosine",
        warmup_frac: float = 0.0,
    ) -> None:
        super().__init__()
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.schedule = schedule
        self.warmup_frac = warmup_frac
        self.register_buffer("_last_momentum", torch.tensor(momentum_start))

    def momentum_at(self, step: int, total_steps: int) -> float:
        """Target momentum at a point in training.

        Every EMA method ramps momentum toward 1: early on the teacher should
        track a rapidly-improving student, later it should be nearly frozen to
        provide a stable target.
        """
        if total_steps <= 1:
            return self.momentum_start
        if self.warmup_frac > 0:
            # data2vec ramps over an initial fraction, then holds.
            frac = min(1.0, step / max(1.0, self.warmup_frac * total_steps))
        else:
            frac = min(1.0, step / (total_steps - 1))

        if self.schedule == "constant":
            return self.momentum_start
        if self.schedule == "linear":
            return self.momentum_start + (self.momentum_end - self.momentum_start) * frac
        if self.schedule == "cosine":
            return (
                self.momentum_end
                - (self.momentum_end - self.momentum_start) * (math.cos(math.pi * frac) + 1) / 2
            )
        raise ValueError(f"unknown EMA schedule {self.schedule!r}")

    @torch.no_grad()
    def update(self, student: nn.Module, step: int, total_steps: int) -> float:
        """One EMA step: ``teacher = m * teacher + (1 - m) * student``."""
        m = self.momentum_at(step, total_steps)
        for t_p, s_p in zip(self.teacher.parameters(), student.parameters(), strict=True):
            t_p.mul_(m).add_(s_p.detach(), alpha=1 - m)
        # Buffers are copied, never averaged -- see the module docstring.
        for t_b, s_b in zip(self.teacher.buffers(), student.buffers(), strict=True):
            t_b.copy_(s_b)
        self._last_momentum.fill_(m)
        return m

    @property
    def last_momentum(self) -> float:
        return float(self._last_momentum)

    @torch.no_grad()
    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.teacher(*args, **kwargs)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        """Delegate to the wrapped module so ``teacher.forward_masked(...)``
        and friends work without re-declaring every encoder method."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["teacher"], name)
