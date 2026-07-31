"""I-JEPA — predict target-block *latents* from a context the encoder never
completes.

This module also carries :class:`JEPABase`, shared with TS-JEPA. The two differ
in exactly one of the ablation's three axes — how the context is constructed —
so the shared base is what makes that claim literal: everything except the mask
geometry is the same code object.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from iqssl.augment.masking import keep_indices
from iqssl.methods.base import Method, cfg_method_arg
from iqssl.models.ema import EMATeacher
from iqssl.models.heads import LatentPredictor
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


class JEPABase(Method):
    """Shared machinery for the JEPA variants: EMA teacher targets at the last
    layer, dropped context, narrow transformer predictor.

    Subclasses supply the mask geometry (``view_spec``) and how to read
    context/target indices out of ``batch.masks``. Nothing else may differ
    between them — the 3-axis ablation is only interpretable if the axes are the
    only variables.
    """

    requires_tokenizer = True

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        mask_ratio: float = 0.5,
        predictor_dim: int = 192,
        predictor_depth: int = 4,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
    ) -> None:
        super().__init__(encoder, cfg)
        self.mask_ratio = mask_ratio
        self.predictor = LatentPredictor(
            self.embed_dim,
            int(encoder.num_patches),  # type: ignore[arg-type]
            predictor_dim=predictor_dim,
            depth=predictor_depth,
        )
        self.teacher = EMATeacher(encoder, momentum_start=momentum_start, momentum_end=momentum_end)

    # -- geometry, supplied by the subclass -----------------------------------

    def context_and_targets(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """``(context_idx, target_idx)``, each ``(B, K)`` int64 token positions."""
        raise NotImplementedError

    # -- shared forward --------------------------------------------------------

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        (x,) = batch.require_views(1, type(self).__name__)
        ctx_idx, tgt_idx = self.context_and_targets(batch)

        # Student: encode the context only. The targets are not merely hidden,
        # they are absent — forward_masked gathers before attention, so target
        # content cannot leak into the context representation.
        latent = self.encoder.forward_masked(x, ctx_idx)  # type: ignore[operator]
        ctx_tokens = latent[:, 1:]  # drop cls; the predictor works on patch tokens

        pred = self.predictor(ctx_tokens, ctx_idx, tgt_idx)

        with torch.no_grad():
            t_tokens = self.teacher(x, return_tokens=True).tokens
            target = torch.gather(
                t_tokens, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, t_tokens.shape[-1])
            )

        loss = F.smooth_l1_loss(pred, target)

        logs = {"loss": float(loss.detach()), "ema_momentum": self.teacher.last_momentum}
        logs.update(self.collapse_logs(ctx_tokens.mean(1), prefix="enc_"))
        # Teacher-target variance is the failure mode to watch: if the teacher's
        # latents collapse, the student hits zero loss while learning nothing,
        # and only this canary says so.
        logs.update(self.collapse_logs(target.flatten(0, 1), prefix="tgt_"))
        return MethodOutput(loss=loss, logs=logs, extras={"pred": pred})

    def on_step_end(self, step: int, total_steps: int) -> None:
        self.teacher.update(self.encoder, step, total_steps)

    def encoder_passes_per_step(self) -> tuple[float, float]:
        # Student sees the context fraction; the teacher runs one full pass.
        return 1.0 - self.mask_ratio, 1.0


@METHODS.register("ijepa")
class IJEPA(JEPABase):
    """Context = everything outside several contiguous target blocks.

    The block structure is the point: on a 1-D signal, randomly scattered
    targets always sit next to a visible token and interpolation solves the
    task. Blocks force genuinely non-local prediction.
    """

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(
            n_views=1,
            needs_mask=True,
            mask_kind="jepa",
            mask_ratio=float(cfg_method_arg(cfg, "mask_ratio", 0.5)),
            symmetry="asymmetric",
        )

    def context_and_targets(self, batch: Batch) -> tuple[Tensor, Tensor]:
        if batch.masks is None or "context" not in batch.masks:
            raise ValueError("IJEPA needs batch.masks['context'/'targets'] (mask_kind='jepa')")
        context = batch.masks["context"]  # True = visible to the student
        union = batch.masks["targets"].any(dim=1)  # True = to be predicted
        ctx_idx, _ = keep_indices(~context)
        tgt_idx, _ = keep_indices(~union)
        return ctx_idx, tgt_idx


@METHODS.register("tsjepa")
class TSJEPA(JEPABase):
    """Context = a contiguous past; targets = the future.

    The time-series variant of the same objective: identical teacher, identical
    predictor, identical loss — only the context construction changes, from
    scattered blocks to a causal split. The gap between this and I-JEPA is
    therefore attributable to context geometry alone.
    """

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        return ViewSpec(
            n_views=1,
            needs_mask=True,
            mask_kind="causal",
            mask_ratio=float(cfg_method_arg(cfg, "mask_ratio", 0.5)),
            symmetry="asymmetric",
        )

    def context_and_targets(self, batch: Batch) -> tuple[Tensor, Tensor]:
        if batch.masks is None or "mask" not in batch.masks:
            raise ValueError("TSJEPA needs batch.masks['mask'] (mask_kind='causal')")
        mask = batch.masks["mask"]  # True = masked suffix
        ctx_idx, _ = keep_indices(mask)
        tgt_idx, _ = keep_indices(~mask)
        return ctx_idx, tgt_idx
