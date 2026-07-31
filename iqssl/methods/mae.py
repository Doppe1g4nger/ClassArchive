"""MAE — masked autoencoding of raw IQ patches."""

from __future__ import annotations

from typing import Any

from torch import nn

from iqssl.augment.masking import keep_indices
from iqssl.methods.base import Method, cfg_method_arg
from iqssl.methods.losses import masked_mse, patchify
from iqssl.models.heads import PatchDecoder
from iqssl.registry import METHODS
from iqssl.types import Batch, MethodOutput, ViewSpec


@METHODS.register("mae")
class MAE(Method):
    """Drop 75% of the tokens, encode the rest, reconstruct what was hidden.

    The efficiency is not an implementation nicety, it is the method: the
    encoder genuinely never sees the masked tokens (``forward_masked`` gathers
    the kept ones before any attention runs), which is what allows the high mask
    ratio that makes the pretext hard enough to be worth solving. Zeroing masked
    positions instead would leak their *locations* through attention and change
    the objective.

    The decoder is deliberately narrow and is discarded after pretraining;
    capacity spent there serves no downstream task.
    """

    requires_tokenizer = True

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Any = None,
        *,
        mask_ratio: float = 0.75,
        decoder_dim: int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = 3,
        normalize_targets: bool = False,
    ) -> None:
        super().__init__(encoder, cfg)
        self.mask_ratio = mask_ratio
        self.normalize_targets = normalize_targets
        self.patch_size = int(encoder.patch_size)  # type: ignore[arg-type]
        self.decoder = PatchDecoder(
            self.embed_dim,
            int(encoder.num_patches),  # type: ignore[arg-type]
            self.patch_size,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
        )

    @classmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        # One view, not zero: the augmentation policy is a sweep axis for the
        # masked family too, and under `none` the view is the raw buffer anyway.
        return ViewSpec(
            n_views=1,
            needs_mask=True,
            mask_kind="random",
            mask_ratio=float(cfg_method_arg(cfg, "mask_ratio", 0.75)),
        )

    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        (x,) = batch.require_views(1, "MAE")
        if batch.masks is None or "mask" not in batch.masks:
            raise ValueError("MAE needs batch.masks['mask']; check the augment pipeline")
        mask = batch.masks["mask"]

        keep_idx, ids_restore = keep_indices(mask)
        latent = self.encoder.forward_masked(x, keep_idx)  # type: ignore[operator]
        pred = self.decoder(latent, ids_restore)

        target = patchify(x, self.patch_size)
        loss, logs = masked_mse(pred, target, mask, normalize_targets=self.normalize_targets)

        # Canaries on the pooled kept-token representation: reconstruction
        # losses cannot collapse the way negative-free ones do, but a dead
        # encoder still shows up here first.
        logs.update(self.collapse_logs(latent[:, 1:].mean(1), prefix="enc_"))
        logs["loss"] = float(loss.detach())
        return MethodOutput(loss=loss, logs=logs, extras={"pred": pred})

    def encoder_passes_per_step(self) -> tuple[float, float]:
        # The encoder runs over the kept fraction only; the narrow decoder is
        # not the encoder and is not billed as one.
        return 1.0 - self.mask_ratio, 0.0
