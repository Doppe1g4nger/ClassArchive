"""VICReg — variance, invariance, covariance."""

from __future__ import annotations

from typing import Any

from torch import nn

from iqssl.methods.base import Method
from iqssl.methods.losses import vicreg
from iqssl.models.heads import Projector
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("vicreg")
class VICReg(Method):
    """Invariance to the view, variance across the batch, decorrelated dimensions.

    The variance hinge is what makes this the most *explicit* anti-collapse
    objective in the comparison: where BYOL and SimSiam prevent collapse through
    an architectural asymmetry whose mechanism is still debated, VICReg simply
    penalises the per-dimension standard deviation falling below 1. Whether an
    explicit term or an implicit asymmetry works better on IQ is one of the
    questions the benchmark exists to answer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        proj_hidden: int = 8192,
        proj_dim: int = 8192,
        pool: str = "mean",
    ) -> None:
        super().__init__(encoder, cfg)
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.pool = pool
        self.projector = Projector(self.embed_dim, proj_hidden, proj_dim)

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(n_views=2, symmetry="symmetric")

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        v1, v2 = batch.require_views(2, "VICReg")
        h1 = self.encoder(v1).pooled(self.pool)
        h2 = self.encoder(v2).pooled(self.pool)
        z1, z2 = self.projector(h1), self.projector(h2)

        loss, logs = vicreg(z1, z2, self.sim_coeff, self.std_coeff, self.cov_coeff)
        logs.update(self.collapse_logs(h1, prefix="enc_"))
        logs["loss"] = float(loss.detach())
        return MethodOutput(loss=loss, logs=logs, extras={"z1": z1, "z2": z2})
