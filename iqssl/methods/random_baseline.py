"""The random floor — a frozen, untrained encoder.

Random features are famously non-trivial (random convolutions beat raw pixels on
most probes), so the floor is not chance and must be *measured*, not assumed.
Every method's headline number is only meaningful as a position between this
floor and the supervised ceiling: a method that beats chance but not random
features has learned nothing worth the compute.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from iqssl.methods.base import Method
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("random")
class RandomBaseline(Method):
    """Pretraining is a no-op; evaluation sees the encoder exactly as built.

    The loop still runs — same steps, same logging, same checkpoint format — so
    the floor flows through the *identical* pipeline as every real method and
    the aggregator needs no special case for it. The encoder is frozen and the
    loss is a constant zero routed through one dummy parameter, because the loop
    is owed a scalar with a graph to backward through and an optimizer is owed
    at least one parameter to hold.
    """

    trainable = False
    """The tiny-overfit suite asserts loss falls for every method; a floor whose
    loss cannot fall is by design, and this flag is how the test knows."""

    def __init__(self, encoder: nn.Module, cfg: Any = None, *, pool: str = "cls") -> None:
        super().__init__(encoder, cfg)
        self.pool = pool
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._dummy = nn.Parameter(torch.zeros(1))

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(n_views=0)

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        with torch.no_grad():
            h = self.encoder(batch.x_raw).pooled(self.pool)
        loss = 0.0 * self._dummy.sum()

        logs = {"loss": float(loss.detach())}
        # Canaries on the frozen features anyway: they are the floor's entire
        # content, and a NaN here means the *encoder init* is broken, which
        # would poison every other method too.
        logs.update(self.collapse_logs(h, prefix="enc_"))
        return MethodOutput(loss=loss, logs=logs)

    def encoder_passes_per_step(self) -> tuple[float, float]:
        return 1.0, 0.0
