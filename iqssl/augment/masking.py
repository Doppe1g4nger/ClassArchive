"""Token masks for the masked-input and latent-prediction methods.

Three kinds, because the three families genuinely need different structure and
collapsing them into one would quietly change what an objective is:

``random``
    Independent per token. MAE and data2vec.
``block``
    Contiguous spans. I-JEPA's targets are blocks, and on a 1-D signal a random
    scatter would leave every masked token with an immediate unmasked neighbour,
    making prediction nearly trivial by interpolation.
``causal``
    Everything after a sampled split point. For the autoregressive ablation.

Masks are drawn per sample from an explicit generator, and ``True`` always means
*masked / to be predicted* -- the convention is stated in :mod:`iqssl.types` and
inverting it somewhere would be a silent, catastrophic sign error.
"""

from __future__ import annotations

import torch
from torch import Generator, Tensor

from iqssl.types import MaskKind


def make_mask(
    batch: int,
    n_tokens: int,
    kind: MaskKind = "random",
    *,
    ratio: float = 0.75,
    block_size: int = 4,
    generator: Generator | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """``(B, N)`` bool, True where masked."""
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask ratio must be in (0, 1), got {ratio}")

    if kind == "random":
        return _random(batch, n_tokens, ratio, generator, device)
    if kind == "block":
        return _block(batch, n_tokens, ratio, block_size, generator, device)
    if kind == "causal":
        return _causal(batch, n_tokens, ratio, generator, device)
    raise ValueError(f"unknown mask kind {kind!r}; options: random, block, causal")


def _random(
    b: int, n: int, ratio: float, g: Generator | None, device: torch.device | str
) -> Tensor:
    """Exactly ``round(ratio*n)`` masked per sample.

    Exactly, not in expectation. Bernoulli masking gives each sample a different
    number of kept tokens, and MAE's encoder gathers a fixed-width tensor of
    them -- a variable count would either ragged the batch or force padding that
    silently becomes a token the model can see.
    """
    k = max(1, min(n - 1, round(ratio * n)))
    noise = torch.rand(b, n, generator=g, device=device)
    idx = noise.argsort(dim=1)
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    return mask.scatter(1, idx[:, :k], True)


def _block(
    b: int, n: int, ratio: float, block_size: int, g: Generator | None, device: torch.device | str
) -> Tensor:
    """Contiguous spans until the target ratio is met.

    Spans may overlap, so the realized ratio is at or slightly above the target
    rather than exact. That is the right trade here: forcing an exact count would
    mean truncating a block, and a half-block is not the structure the objective
    is meant to see.
    """
    target = max(1, min(n - 1, round(ratio * n)))
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    width = max(1, min(block_size, n - 1))
    idx = torch.arange(n, device=device).unsqueeze(0)

    # Bounded rather than `while`: with overlapping spans the count converges
    # quickly, but an unlucky draw must not hang a training run.
    for _ in range(4 * (target // width + 1)):
        if int(mask.sum(1).min()) >= target:
            break
        start = torch.randint(0, max(1, n - width + 1), (b,), generator=g, device=device)
        span = (idx >= start.unsqueeze(-1)) & (idx < (start + width).unsqueeze(-1))
        # Only extend samples that are still short of the target.
        needs = (mask.sum(1) < target).unsqueeze(-1)
        mask = mask | (span & needs)
    return mask


def _causal(
    b: int, n: int, ratio: float, g: Generator | None, device: torch.device | str
) -> Tensor:
    """Mask a suffix. The split point jitters around ``1 - ratio``."""
    centre = (1.0 - ratio) * n
    jitter = (torch.rand(b, generator=g, device=device) - 0.5) * 0.2 * n
    split = (centre + jitter).clamp(1, n - 1).long()
    idx = torch.arange(n, device=device).unsqueeze(0)
    return idx >= split.unsqueeze(-1)


def keep_indices(mask: Tensor) -> tuple[Tensor, Tensor]:
    """``(keep_idx, ids_restore)`` for MAE's gather/scatter path.

    ``keep_idx`` is ``(B, N_keep)`` int64 into the token axis; ``ids_restore``
    un-shuffles the decoder's concatenated [kept | mask tokens] back to input
    order. Requires every row of ``mask`` to have the same count, which is why
    :func:`_random` masks an exact number.
    """
    counts = (~mask).sum(1)
    if int(counts.min()) != int(counts.max()):
        raise ValueError(
            f"keep_indices needs a uniform keep count per sample, got "
            f"{int(counts.min())}..{int(counts.max())}; use mask_kind='random'"
        )
    order = mask.long().argsort(dim=1, stable=True)  # False (kept) first
    ids_restore = order.argsort(dim=1)
    return order[:, : int(counts[0])], ids_restore
