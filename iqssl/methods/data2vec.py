"""data2vec — regress an EMA teacher's *layer-averaged* latents at masked
positions.

The third point on the latent-prediction ablation. Against the JEPA pair it
varies two axes at once by design of the original method: targets come from the
average of the top-K teacher layers rather than the last layer alone, and the
context is the full-length sequence with mask tokens substituted in place rather
than dropped. The predictor is a linear head — the lightest of the three.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from iqssl.methods.base import Method, cfg_method_arg
from iqssl.methods.losses import masked_smooth_l1
from iqssl.models.ema import EMATeacher
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("data2vec")
class Data2Vec(Method):
    """Masked latent regression with layer-averaged targets.

    Two details are the method, and both are collapse defenses:

    **Targets average the top-K layers, each normalized first.** A single-layer
    target lets the teacher drift toward a low-variance representation the
    student can match trivially; averaging over depth — with per-token
    normalization *before* the average so no one layer's scale dominates —
    empirically stabilizes the target distribution. K and the normalization are
    not tuning freedom, they are the published recipe.

    **The EMA ramps fast and then holds.** data2vec's schedule reaches its final
    momentum early (`warmup_frac`), unlike BYOL's full-run cosine; a teacher
    that keeps accelerating away from the student late in training destabilizes
    the regression targets.
    """

    requires_tokenizer = True

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        mask_ratio: float = 0.5,
        top_k_layers: int = 4,
        beta: float = 1.0,
        momentum_start: float = 0.999,
        momentum_end: float = 0.9999,
        ema_warmup_frac: float = 0.05,
    ) -> None:
        super().__init__(encoder, cfg)
        self.mask_ratio = mask_ratio
        self.top_k = top_k_layers
        self.beta = beta
        self.head = nn.Linear(self.embed_dim, self.embed_dim)
        self.teacher = EMATeacher(
            encoder,
            momentum_start=momentum_start,
            momentum_end=momentum_end,
            schedule="linear",
            warmup_frac=ema_warmup_frac,
        )

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        # Block masking, per the paper: scattered single-token masks on a 1-D
        # signal are solvable by interpolation from immediate neighbours.
        return ViewSpec(
            n_views=1,
            needs_mask=True,
            mask_kind="block",
            mask_ratio=float(cfg_method_arg(cfg, "mask_ratio", 0.5)),
            mask_block_size=int(cfg_method_arg(cfg, "mask_block_size", 4)),
            symmetry="asymmetric",
        )

    def _targets(self, x: Tensor) -> Tensor:
        """Top-K layer average from the teacher, per-token normalized."""
        with torch.no_grad():
            out = self.teacher(x, return_all_layers=True)
            layers = out.layers
            assert layers is not None
            k = min(self.top_k, len(layers))
            stacked = torch.stack([F.layer_norm(layer, layer.shape[-1:]) for layer in layers[-k:]])
            return stacked.mean(0)

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        (x,) = batch.require_views(1, "Data2Vec")
        if batch.masks is None or "mask" not in batch.masks:
            raise ValueError("Data2Vec needs batch.masks['mask']; check the augment pipeline")
        mask = batch.masks["mask"]

        # Student sees the full-length sequence with mask tokens substituted in
        # place — masked positions still *attend*, they just carry no content.
        # That is the context-construction axis this method occupies.
        tokens = self.encoder.forward_with_mask_token(x, mask)  # type: ignore[operator]
        pred = self.head(tokens)

        target = self._targets(x)
        loss = masked_smooth_l1(pred, target, mask, beta=self.beta)

        logs = {"loss": float(loss.detach()), "ema_momentum": self.teacher.last_momentum}
        logs.update(self.collapse_logs(tokens.mean(1), prefix="enc_"))
        # Target collapse is data2vec's characteristic failure: teacher latents
        # shrink toward a constant, student loss falls to zero, nothing was
        # learned. The tgt_ canaries are the only early warning.
        logs.update(self.collapse_logs(target.flatten(0, 1), prefix="tgt_"))
        return MethodOutput(loss=loss, logs=logs, extras={"pred": pred})

    def on_step_end(self, step: int, total_steps: int) -> None:
        self.teacher.update(self.encoder, step, total_steps)

    def encoder_passes_per_step(self) -> tuple[float, float]:
        # Full-length student pass (mask tokens still attend), one teacher pass.
        return 1.0, 1.0
