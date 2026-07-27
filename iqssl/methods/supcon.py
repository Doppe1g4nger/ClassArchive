"""SupCon — the supervised control for the contrastive family.

Not a competitor but a *reference point*: identical encoder, identical
augmentations, identical projector as SimCLR, differing only in which pairs
count as positive. The gap between the two is therefore attributable to label
information alone, which makes it the natural ceiling for how much a contrastive
objective could extract if it knew the answer.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from iqssl.methods.base import Method
from iqssl.methods.losses import supcon
from iqssl.models.heads import Projector
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("supcon")
class SupCon(Method):
    """Supervised contrastive learning on the primary label."""

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
        # needs_labels also switches the loader to a class-balanced sampler:
        # with many classes and a modest batch, a uniform sampler leaves most
        # anchors with no positives at all and the loss degenerates toward
        # NT-Xent without anyone noticing.
        return ViewSpec(n_views=2, needs_labels=True, symmetry="symmetric")

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        v1, v2 = batch.require_views(2, "SupCon")
        labels = batch.require_primary("SupCon")

        h1 = self.encoder(v1).pooled(self.pool)
        h2 = self.encoder(v2).pooled(self.pool)
        z1, z2 = self.projector(h1), self.projector(h2)

        loss, logs = supcon(z1, z2, labels, self.temperature)
        logs.update(self.collapse_logs(h1, prefix="enc_"))
        logs["loss"] = float(loss.detach())
        return MethodOutput(loss=loss, logs=logs, extras={"z1": z1, "z2": z2})
