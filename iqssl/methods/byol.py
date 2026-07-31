"""BYOL — bootstrap your own latent, via an EMA teacher."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from iqssl.methods.base import Method, default_param_groups
from iqssl.methods.losses import negative_cosine
from iqssl.models.ema import EMATeacher
from iqssl.models.heads import Predictor, Projector
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("byol")
class BYOL(Method):
    """Predict the EMA teacher's projection of the other view.

    Two asymmetries prevent collapse, and both must be present: the online branch
    carries an extra predictor the target branch lacks, and the target branch is
    a momentum copy rather than the student itself. Remove either and the trivial
    constant solution becomes reachable.

    The loss is symmetrized -- each view predicts the other -- which is what the
    paper does and roughly halves the variance per step. The ``ViewSpec`` still
    declares ``asymmetric`` because view 0 and view 1 pass through *different*
    networks, which is what the flag describes.
    """

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        proj_hidden: int = 4096,
        proj_dim: int = 256,
        pred_hidden: int = 4096,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
        predictor_lr_scale: float = 10.0,
        pool: str = "cls",
    ) -> None:
        super().__init__(encoder, cfg)
        self.pool = pool
        self.predictor_lr_scale = predictor_lr_scale

        self.projector = Projector(self.embed_dim, proj_hidden, proj_dim)
        self.predictor = Predictor(proj_dim, pred_hidden, proj_dim)

        # Encoder and projector are wrapped together so the teacher is a momentum
        # copy of *both*. The student's target is a projection, so a teacher that
        # shared the student's live projector would be chasing a target that
        # moves with every step, and the asymmetry would be only half present.
        self.online = nn.ModuleDict({"encoder": self.encoder, "projector": self.projector})
        self.teacher = EMATeacher(
            self.online,
            momentum_start=momentum_start,
            momentum_end=momentum_end,
        )

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(n_views=2, symmetry="asymmetric")

    def _online(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.projector(self.encoder(x).pooled(self.pool)))

    @torch.no_grad()
    def _target(self, x: torch.Tensor) -> torch.Tensor:
        t = cast(nn.ModuleDict, self.teacher.teacher)
        return t["projector"](t["encoder"](x).pooled(self.pool))

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        v1, v2 = batch.require_views(2, "BYOL")
        p1, p2 = self._online(v1), self._online(v2)
        z1, z2 = self._target(v1), self._target(v2)

        # Detached at the call site rather than inside the loss, so the
        # stop-gradient stays visible in the method that depends on it.
        loss = 0.5 * (negative_cosine(p1, z2.detach()) + negative_cosine(p2, z1.detach()))

        logs = {"loss": float(loss.detach()), "ema_momentum": self.teacher.last_momentum}
        logs.update(self.collapse_logs(p1, prefix="pred_"))
        return MethodOutput(loss=loss, logs=logs, extras={"p1": p1, "z1": z1})

    def on_step_end(self, step: int, total_steps: int) -> None:
        self.teacher.update(self.online, step, total_steps)

    def encoder_passes_per_step(self) -> tuple[float, float]:
        # Symmetrized loss: both views pass through both branches.
        return 2.0, 2.0

    def param_groups(self, base_lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Predictor at 10x LR, everything else at the base rate.

        The teacher's parameters are excluded entirely -- they are updated by EMA,
        and handing them to the optimizer would have gradient descent and the
        momentum update fighting over the same tensors.
        """
        groups = default_param_groups(
            nn.ModuleDict({"encoder": self.encoder, "projector": self.projector}),
            base_lr,
            weight_decay,
        )
        for g in default_param_groups(self.predictor, base_lr, weight_decay):
            g["lr_scale"] = self.predictor_lr_scale
            groups.append(g)
        return groups

    def encoder_for_eval(self) -> nn.Module:
        """The student, matching the paper.

        The teacher is a valid alternative and sometimes scores slightly higher,
        so the choice is recorded here rather than left implicit -- reporting
        whichever happened to win would be a quiet degree of freedom.
        """
        return self.encoder
