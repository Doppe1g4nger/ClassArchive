"""The Method contract.

Every SSL objective in this repository implements exactly this interface, and
the training loop knows nothing else about them. That is the mechanism by which
the benchmark is fair: if the loop had to special-case SimCLR's two views or
MAE's masking, each special case would be an opportunity for one method to get a
scheduling or batching advantage the others never received.

Four hooks cover everything the eleven objectives need:

``view_spec``
    A *classmethod*, so the collate function can be built before the model
    exists and so tests can enumerate the contract without constructing
    anything.
``forward``
    Receives the global step and total steps, because several methods ramp
    coefficients or EMA momentum over training and must not have to track that
    state themselves.
``on_step_end``
    Where EMA teachers update. Separate from ``forward`` so it happens *after*
    the optimizer step, which is what the published algorithms specify.
``param_groups``
    Lets a method request per-parameter learning-rate treatment — BYOL's 10x
    predictor, SimSiam's predictor held out of cosine decay — without the
    optimizer needing to know which method it is serving.
"""

from __future__ import annotations

import abc
from typing import Any

import torch
from torch import Tensor, nn

from iqssl.types import Batch, MethodOutput, ViewSpec


class Method(nn.Module, abc.ABC):
    """Base class for every self-supervised objective."""

    #: Set False by methods that need patch tokens (MAE, JEPA, data2vec).
    #: Checked against the encoder at config-validation time.
    requires_tokenizer: bool = False

    def __init__(self, encoder: nn.Module, cfg: Any = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        if self.requires_tokenizer and not getattr(encoder, "supports_masking", False):
            raise ValueError(
                f"{type(self).__name__} needs an encoder with patch tokens "
                f"(encoder=vit1d), but got {type(encoder).__name__}. Masked "
                "convolutions are a different problem; faking token dropping by "
                "zeroing the input would silently change the objective."
            )

    # -- contract -------------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def view_spec(cls, cfg: Any = None) -> ViewSpec:
        """What this method needs a batch to contain."""

    @abc.abstractmethod
    def forward(self, batch: Batch, step: int, total_steps: int) -> MethodOutput:
        """Compute the loss for one batch."""

    def on_step_end(self, step: int, total_steps: int) -> None:
        """Called after the optimizer step. EMA updates belong here."""

    def param_groups(self, base_lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Optimizer parameter groups.

        The default applies the near-universal SSL convention: no weight decay
        on any 1-D parameter (biases, norm weights, class and mask tokens,
        positional tables). Decaying those measurably hurts and is omitted in
        essentially every published implementation.

        Methods override to add ``lr_scale`` (BYOL's 10x predictor) or
        ``fix_lr`` (SimSiam's predictor, held out of cosine decay).
        """
        return default_param_groups(self, base_lr, weight_decay)

    def encoder_passes_per_step(self) -> tuple[float, float]:
        """``(student_passes, teacher_passes)`` per training step, token-weighted.

        The fairness argument holds epochs constant, but equal epochs is not
        equal compute: MAE's encoder sees 25% of the tokens, two-view methods do
        2x the passes, and EMA methods add gradient-free teacher passes. The
        Pareto plot in the analysis stage is built from these numbers, so a
        method that misdeclares them is quietly flattering itself.

        The default charges one full-token student pass per view (minimum one)
        and no teacher. Methods override: MAE reports its keep-fraction, the
        EMA methods report their teacher passes. Declared rather than measured
        because the loop must stay method-agnostic — and a declaration sitting
        next to the forward that spends it is easy to audit.
        """
        return float(max(1, type(self).view_spec(self.cfg).n_views)), 0.0

    def encoder_for_eval(self) -> nn.Module:
        """Which encoder evaluation should use.

        Defaults to the student. EMA-teacher methods may override — but should
        say so in the results, since teacher and student features can differ
        materially.
        """
        return self.encoder

    @property
    def embed_dim(self) -> int:
        """Encoder feature width.

        Goes through an explicit cast because ``nn.Module.__getattr__`` is typed
        as returning ``Tensor | Module``, so reading ``encoder.embed_dim``
        directly makes every head construction a type error.
        """
        return int(self.encoder.embed_dim)  # type: ignore[arg-type]

    # -- shared helpers -------------------------------------------------------

    @staticmethod
    def collapse_logs(z: Tensor, prefix: str = "") -> dict[str, float]:
        """Collapse canary, logged every step by every method.

        Representational collapse is the characteristic failure of negative-free
        and latent-prediction objectives, and it is *silent*: the loss falls
        beautifully while the encoder maps everything to one point. Per-dimension
        standard deviation and effective rank both fall off a cliff when it
        happens, so they are cheap early warnings.
        """
        with torch.no_grad():
            zf = z.detach().float()
            std = zf.std(0)
            return {
                f"{prefix}std_mean": float(std.mean()),
                f"{prefix}std_min": float(std.min()),
                f"{prefix}dead_dims": float((std < 1e-3).float().mean()),
                f"{prefix}rankme": float(rankme(zf)),
            }


def cfg_method_arg(cfg: Any, name: str, default: Any) -> Any:
    """Read ``cfg.method.args.<name>``, tolerating a missing or None cfg.

    Exists for the masked methods' ``view_spec`` classmethods: mask geometry
    lives in the ViewSpec, the ViewSpec is built before the method instance, and
    the only configuration available at that point is the raw Hydra tree. This
    is the one sanctioned way to reach into it, so the instance argument and the
    spec cannot come from two different sources and silently disagree.
    """
    try:
        args = cfg.method.args
    except AttributeError:
        return default
    try:
        value = args.get(name, default)
    except AttributeError:
        return default
    return default if value is None else value


def rankme(z: Tensor, eps: float = 1e-7) -> Tensor:
    """Effective rank: ``exp(H(sigma / sum(sigma)))`` over singular values.

    A smooth, differentiable-free stand-in for "how many directions does this
    representation actually use". Correlates with downstream accuracy well
    enough to be a useful training-time diagnostic, and unlike a hard rank it
    degrades gracefully.
    """
    if z.ndim != 2 or min(z.shape) < 2:
        return torch.tensor(float("nan"))
    sv = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    p = sv / (sv.sum() + eps)
    return torch.exp(-(p * torch.log(p + eps)).sum())


def default_param_groups(
    module: nn.Module, base_lr: float, weight_decay: float
) -> list[dict[str, Any]]:
    """Split parameters into decayed and non-decayed groups."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        # 1-D parameters are biases, norm gains, and learned tokens.
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = []
    if decay:
        groups.append(
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay, "lars_exclude": False}
        )
    if no_decay:
        groups.append(
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0, "lars_exclude": True}
        )
    return groups
