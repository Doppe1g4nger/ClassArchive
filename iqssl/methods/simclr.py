"""SimCLR — contrastive learning with in-batch negatives."""

from __future__ import annotations

from typing import Any

from torch import nn

from iqssl.methods.base import Method
from iqssl.methods.losses import nt_xent
from iqssl.models.heads import Projector
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("simclr")
class SimCLR(Method):
    """Two augmented views; each view's counterpart is its only positive.

    Projector is 2048-2048-128. The narrow output is deliberate and is *not* a
    fairness problem to be equalized away: contrastive objectives work best in a
    low-dimensional normalized space, while Barlow Twins and VICReg need wide
    projectors. The encoder is the held-fixed control variable; heads are part
    of the method.
    """

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        temperature: float = 0.1,
        proj_hidden: int = 2048,
        proj_dim: int = 128,
        pool: str = "cls",
    ) -> None:
        super().__init__(encoder, cfg)
        self.temperature = temperature
        self.pool = pool
        self.projector = Projector(self.embed_dim, proj_hidden, proj_dim)

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(n_views=2, symmetry="symmetric")

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        v1, v2 = batch.require_views(2, "SimCLR")
        h1 = self.encoder(v1).pooled(self.pool)
        h2 = self.encoder(v2).pooled(self.pool)
        z1, z2 = self.projector(h1), self.projector(h2)

        loss, logs = nt_xent(z1, z2, self.temperature)
        logs.update(self.collapse_logs(h1, prefix="enc_"))
        logs["loss"] = float(loss.detach())
        return MethodOutput(loss=loss, logs=logs, extras={"z1": z1, "z2": z2})
