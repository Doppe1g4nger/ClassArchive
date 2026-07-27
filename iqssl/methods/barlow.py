"""Barlow Twins — redundancy reduction via the cross-correlation matrix."""

from __future__ import annotations

from typing import Any

from torch import nn

from iqssl.methods.base import Method
from iqssl.methods.losses import barlow_twins
from iqssl.models.heads import Projector
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("barlow")
class BarlowTwins(Method):
    """Push the cross-correlation between two views toward the identity.

    No negatives and no stop-gradient: collapse is prevented by the off-diagonal
    penalty alone, which is what makes it an interesting contrast with the
    distillation methods that need an asymmetry to achieve the same thing.

    The 8192-d projector is load-bearing rather than incidental -- the objective
    decorrelates *dimensions*, so it has more to work with the wider it is, and
    the published results degrade sharply when it is narrowed. Capping it at
    SimCLR's 128 in the name of fairness would be a handicap dressed up as one.
    """

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        lambd: float = 5e-3,
        proj_hidden: int = 8192,
        proj_dim: int = 8192,
        pool: str = "mean",
    ) -> None:
        super().__init__(encoder, cfg)
        self.lambd = lambd
        self.pool = pool
        self.projector = Projector(self.embed_dim, proj_hidden, proj_dim)

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(n_views=2, symmetry="symmetric")

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        v1, v2 = batch.require_views(2, "BarlowTwins")
        h1 = self.encoder(v1).pooled(self.pool)
        h2 = self.encoder(v2).pooled(self.pool)
        z1, z2 = self.projector(h1), self.projector(h2)

        loss, logs = barlow_twins(z1, z2, self.lambd)
        logs.update(self.collapse_logs(h1, prefix="enc_"))
        logs["loss"] = float(loss.detach())
        return MethodOutput(loss=loss, logs=logs, extras={"z1": z1, "z2": z2})
