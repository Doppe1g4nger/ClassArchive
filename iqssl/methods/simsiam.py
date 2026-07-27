"""SimSiam — negative-free, teacher-free, held up by a stop-gradient alone."""

from __future__ import annotations

from typing import Any

from torch import nn

from iqssl.methods.base import Method, default_param_groups
from iqssl.methods.losses import negative_cosine
from iqssl.models.heads import Predictor, Projector
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("simsiam")
class SimSiam(Method):
    """BYOL with the momentum teacher removed.

    That makes it the sharpest control in the negative-free family: it shares
    BYOL's predictor asymmetry but has no EMA at all, so the gap between the two
    isolates what the momentum encoder actually contributes. If SimSiam matches
    BYOL here, the EMA is doing nothing on IQ that the stop-gradient does not
    already do.

    Two details are genuinely load-bearing and both are easy to lose in a
    refactor: the stop-gradient on the target branch, and holding the predictor's
    learning rate constant. Removing either collapses the representation, and it
    collapses *silently* -- the loss falls beautifully while every input maps to
    one point, which is why ``collapse_logs`` is emitted every step.
    """

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        proj_hidden: int = 2048,
        proj_dim: int = 2048,
        pred_hidden: int = 512,
        pool: str = "cls",
    ) -> None:
        super().__init__(encoder, cfg)
        self.pool = pool
        self.projector = Projector(self.embed_dim, proj_hidden, proj_dim)
        # The predictor bottleneck (2048 -> 512 -> 2048) is part of the method,
        # not an arbitrary width: widening it to match the projector measurably
        # degrades the representation in the original ablation.
        self.predictor = Predictor(proj_dim, pred_hidden, proj_dim)

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(n_views=2, symmetry="asymmetric")

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        v1, v2 = batch.require_views(2, "SimSiam")
        z1 = self.projector(self.encoder(v1).pooled(self.pool))
        z2 = self.projector(self.encoder(v2).pooled(self.pool))
        p1, p2 = self.predictor(z1), self.predictor(z2)

        # `.detach()` here is the entire anti-collapse mechanism. It stays at the
        # call site, visible in the method, rather than hidden inside
        # negative_cosine -- one line, and the difference between a working
        # objective and a constant function.
        loss = 0.5 * (negative_cosine(p1, z2.detach()) + negative_cosine(p2, z1.detach()))

        logs = {"loss": float(loss.detach())}
        logs.update(self.collapse_logs(z1, prefix="proj_"))
        return MethodOutput(loss=loss, logs=logs, extras={"z1": z1, "p1": p1})

    def param_groups(self, base_lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Predictor held out of the cosine decay.

        ``fix_lr`` is honoured by :func:`iqssl.train.schedules.apply_lr`. This is
        not a tuning preference -- the SimSiam paper ablates it directly, and a
        decayed predictor loses several points of accuracy.
        """
        groups = default_param_groups(
            nn.ModuleDict({"encoder": self.encoder, "projector": self.projector}),
            base_lr,
            weight_decay,
        )
        for g in default_param_groups(self.predictor, base_lr, weight_decay):
            g["fix_lr"] = True
            groups.append(g)
        return groups
