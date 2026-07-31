"""The supervised ceiling — plain cross-entropy on the primary label."""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import nn

from iqssl.methods.base import Method
from iqssl.models.heads import LinearClassifier
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("supervised")
class Supervised(Method):
    """What full label access buys, through the *same* encoder, loop and budget.

    This is not the difficulty gate's oracle: SmallCNN answers "is the task
    learnable at all?" with a deliberately unremarkable model, while this run
    answers "what should an SSL method aspire to?" under exactly the conditions
    the SSL methods get — same encoder, same schedule, same augmentation policy,
    same compute ledger. Comparing an SSL score against a ceiling trained under
    different conditions would smuggle those conditions into the gap.

    ``n_classes`` must be known at construction — a head built lazily on the
    first batch would come into existence *after* ``param_groups()`` handed the
    optimizer its parameters, and would silently never train. The CLI fills it
    from ``dataset.num_primary_classes``; the default only has to cover the test
    fixtures.
    """

    def __init__(
        self, encoder: nn.Module, cfg: Any = None, *, n_classes: int = 16, pool: str = "cls"
    ) -> None:
        super().__init__(encoder, cfg)
        self.pool = pool
        # use_bn=False: the affine-free BN in LinearClassifier is the *frozen*-
        # feature probe protocol; here the encoder trains end-to-end.
        self.classifier = LinearClassifier(self.embed_dim, n_classes, use_bn=False)

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        # One augmented view — the ceiling trains under the same channel
        # augmentation the SSL methods see. Handing it the raw buffer instead
        # would give it a different (easier) training distribution, and the
        # SSL-to-ceiling gap would partly measure that instead of labels.
        return ViewSpec(n_views=1, needs_labels=True)

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        (x,) = batch.require_views(1, "Supervised")
        y = batch.require_primary("Supervised")

        h = self.encoder(x).pooled(self.pool)
        logits = self.classifier(h)
        loss = F.cross_entropy(logits, y)

        logs = {
            "loss": float(loss.detach()),
            "train_acc": float((logits.argmax(-1) == y).float().mean()),
        }
        logs.update(self.collapse_logs(h, prefix="enc_"))
        return MethodOutput(loss=loss, logs=logs)
