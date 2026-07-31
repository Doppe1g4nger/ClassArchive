"""Frozen-feature extraction.

Deterministic on purpose: crops are centred (``random_crop=False``) and the
encoder is in eval mode, so extracting twice gives identical features and any
probe-to-probe variance is attributable to the probe, not to the pipeline.

Pooling is a protocol constant, the same for every method. The masked family
tends to probe better on mean-pooled tokens and the contrastive family trains
its cls token — letting each method pick its favourite would fold a per-method
choice into what claims to be one protocol. ``mean`` is defined for both
encoders (ResNet1D's ``cls`` *is* its global average), so it is the one that
privileges nobody.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from iqssl.data.dataset import IQDataset

POOL = "mean"
EXTRACT_BATCH = 256


@dataclass
class SplitFeatures:
    """Everything the probes need from one split, all aligned by row."""

    z: Tensor
    """(N, D) float32 frozen features."""

    y_emitter: np.ndarray
    y_mod: np.ndarray
    nuisance: np.ndarray
    """(N, K) standardized with train-split statistics (stored at build time)."""

    snr: np.ndarray

    def labels(self, axis: str) -> np.ndarray:
        if axis == "emitter":
            return self.y_emitter
        if axis == "modulation":
            return self.y_mod
        raise ValueError(f"unknown label axis {axis!r}; options: emitter, modulation")


@torch.no_grad()
def extract(
    encoder: nn.Module, dataset: IQDataset, device: str = "cpu", limit: int | None = None
) -> SplitFeatures:
    """Frozen features for a split, optionally over its first ``limit`` rows.

    Truncation rather than sampling: a split's row order is already a fixed,
    seed-derived permutation, so a prefix is an unbiased subset that needs no
    RNG of its own and is identical across every caller. Hyperparameter search
    uses it to keep per-trial cost bounded; the reported protocol never does.
    """
    dev = torch.device(device)
    encoder = encoder.to(dev).eval()

    n = len(dataset) if limit is None else min(limit, len(dataset))
    feats: list[Tensor] = []
    y_em: list[int] = []
    y_mod: list[int] = []
    nz: list[np.ndarray] = []
    for start in range(0, n, EXTRACT_BATCH):
        items = [dataset[i] for i in range(start, min(start + EXTRACT_BATCH, n))]
        x = torch.stack([it["x"] for it in items]).to(dev)
        feats.append(encoder(x).pooled(POOL).cpu())
        y_em += [it["y_emitter"] for it in items]
        y_mod += [it["y_mod"] for it in items]
        nz += [it["nuisance"].numpy() for it in items]

    return SplitFeatures(
        z=torch.cat(feats),
        y_emitter=np.asarray(y_em, dtype=np.int64),
        y_mod=np.asarray(y_mod, dtype=np.int64),
        nuisance=np.stack(nz),
        snr=dataset.snr[:n].copy(),
    )
