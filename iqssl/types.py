"""Core contracts shared by the dataset, the training loop and every method.

These three dataclasses are what make the eleven SSL objectives interchangeable.
The training loop reads :class:`ViewSpec` to decide how to build a batch, hands
the method a :class:`Batch`, and gets back a :class:`MethodOutput`. It never
branches on which method it is training.

Tensor conventions, decided once and enforced everywhere:

===================  ===========  =========================================
stage                dtype        shape
===================  ===========  =========================================
``iqssl.dsp``        complex64    ``(B, L)``
on-disk shards       float32      ``(N, 2, L_store)``
model input          float32      ``(B, 2, L)``, channel 0 = I, 1 = Q
masks                bool         ``(B, N_tokens)``, True = masked/target
keep indices         int64        ``(B, N_keep)``
labels               int64        ``(B,)``
nuisance             float32      ``(B, K)``, standardized w/ train stats
===================  ===========  =========================================

Complex tensors never reach a model: ``conv1d`` rejects them and complex
autograd is a hazard we have no reason to take on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

MaskKind = Literal["random", "block", "causal", "jepa"]
Pool = Literal["cls", "mean"]


@dataclass(frozen=True)
class ViewSpec:
    """What a method needs a batch to contain.

    Declared by the method *class* (not an instance) so the collate function can
    be built before the model is, and so tests can enumerate the contract
    without constructing anything.
    """

    n_views: int = 2
    """Number of independently augmented views. 0 means the method consumes only
    the unaugmented buffer (e.g. MAE/JEPA under the ``augment=none`` policy)."""

    needs_mask: bool = False
    mask_kind: MaskKind | None = None
    needs_labels: bool = False
    """SupCon and the supervised baseline. Enables the class-balanced sampler."""

    needs_unaugmented: bool = False
    """MAE reconstructs toward the clean buffer when ``target_source='raw'``."""

    symmetry: Literal["symmetric", "asymmetric"] = "symmetric"
    """Asymmetric methods (BYOL, SimSiam, JEPA) treat view 0 and view 1
    differently; symmetric ones may swap them freely."""

    mask_ratio: float = 0.75
    """Fraction of tokens masked (for ``jepa``, the fraction covered by target
    blocks). Lives here rather than in the pipeline because the geometry is part
    of the *objective*: MAE at 75% random and data2vec at 50% block are different
    methods, not different data settings, and a pipeline-level knob would let an
    experiment change one method's objective while claiming to hold it fixed."""

    mask_block_size: int = 4
    """Contiguous span, in tokens, for ``block`` and ``jepa`` masks."""

    def __post_init__(self) -> None:
        if self.n_views < 0:
            raise ValueError(f"n_views must be >= 0, got {self.n_views}")
        if self.needs_mask and self.mask_kind is None:
            raise ValueError("needs_mask=True requires an explicit mask_kind")
        if not self.needs_mask and self.mask_kind is not None:
            raise ValueError(f"mask_kind={self.mask_kind!r} set but needs_mask=False")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")
        if self.mask_block_size < 1:
            raise ValueError(f"mask_block_size must be >= 1, got {self.mask_block_size}")


@dataclass
class Batch:
    """One training batch.

    ``views`` is empty for methods with ``n_views == 0``; ``x_raw`` is always
    populated so eval and reconstruction targets have a canonical input.
    """

    x_raw: Tensor
    """(B, 2, L) float32 — cropped, normalized, unaugmented."""

    views: list[Tensor] = field(default_factory=list)
    """Each (B, 2, L) float32."""

    y_mod: Tensor | None = None
    y_emitter: Tensor | None = None
    y_primary: Tensor | None = None
    """(B,) int64 — an alias of y_emitter or y_mod per ``data.primary_label``."""

    nuisance: Tensor | None = None
    """(B, K) float32, standardized with train-split statistics."""

    nuisance_names: tuple[str, ...] = ()

    masks: dict[str, Tensor] | None = None
    """Method-specific. I-JEPA uses ``{"context": (B,N) bool, "targets": (B,T,N) bool}``;
    MAE and data2vec use ``{"mask": (B,N) bool}`` with True = masked."""

    index: Tensor | None = None
    """(B,) int64 — row index into the dataset, for provenance and debugging."""

    @property
    def batch_size(self) -> int:
        return int(self.x_raw.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.x_raw.shape[-1])

    def to(self, device: torch.device | str, non_blocking: bool = False) -> Batch:
        """Move every tensor field to ``device``, leaving metadata alone."""

        def mv(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.to(device, non_blocking=non_blocking)

        return Batch(
            x_raw=self.x_raw.to(device, non_blocking=non_blocking),
            views=[v.to(device, non_blocking=non_blocking) for v in self.views],
            y_mod=mv(self.y_mod),
            y_emitter=mv(self.y_emitter),
            y_primary=mv(self.y_primary),
            nuisance=mv(self.nuisance),
            nuisance_names=self.nuisance_names,
            masks=(
                None
                if self.masks is None
                else {k: v.to(device, non_blocking=non_blocking) for k, v in self.masks.items()}
            ),
            index=mv(self.index),
        )

    def require_views(self, n: int, who: str) -> list[Tensor]:
        """Fetch exactly ``n`` views or raise with a message naming the caller.

        Guards against a collate/ViewSpec mismatch surfacing as an opaque
        IndexError deep inside a loss function.
        """
        if len(self.views) < n:
            raise ValueError(
                f"{who} needs {n} views but the batch has {len(self.views)}; "
                f"check {who}.view_spec() against the configured augment policy"
            )
        return self.views[:n]

    def require_primary(self, who: str) -> Tensor:
        if self.y_primary is None:
            raise ValueError(f"{who} needs labels but the batch has none (needs_labels=False?)")
        return self.y_primary


@dataclass
class MethodOutput:
    """What every method returns from ``forward``."""

    loss: Tensor
    """Scalar, requires_grad."""

    logs: dict[str, float] = field(default_factory=dict)
    """Scalars for CSV/TensorBoard. Include a collapse canary (per-dim std)."""

    extras: dict[str, Tensor] = field(default_factory=dict)
    """Tensors a caller may want (embeddings for diagnostics) but that are not logged."""

    def __post_init__(self) -> None:
        if self.loss.ndim != 0:
            raise ValueError(f"loss must be a scalar, got shape {tuple(self.loss.shape)}")
