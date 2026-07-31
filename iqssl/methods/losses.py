"""Shared loss functions.

Kept in one place so the numerically delicate parts are written once and tested
once. Each function documents the specific mistake it is written to avoid —
these are the details that separate a working implementation from one that
trains, converges, and reports a number 5 points too low.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

NEG_INF = float("-inf")


def nt_xent(z1: Tensor, z2: Tensor, temperature: float = 0.1) -> tuple[Tensor, dict[str, float]]:
    """SimCLR's normalized temperature-scaled cross entropy.

    Two views of ``B`` samples are concatenated into ``2B`` embeddings; each
    embedding's positive is its counterpart and every other embedding is a
    negative.

    Two details matter:

    * Self-similarities are masked with ``-inf``, not ``0``. Zero is a perfectly
      plausible logit, so masking with it leaves each anchor competing against a
      phantom negative of moderate similarity — a small, systematic bias.
    * The row max is subtracted before the softmax. ``logsumexp`` handles this
      internally, which is why it is used here rather than a manual ``exp/sum``.
    """
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    sim = z @ z.t() / temperature

    eye = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, NEG_INF)

    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    loss = F.cross_entropy(sim, targets)

    with torch.no_grad():
        acc = float((sim.argmax(-1) == targets).float().mean())
        pos_sim = float((F.normalize(z1, dim=-1) * F.normalize(z2, dim=-1)).sum(-1).mean())
    return loss, {"contrastive_acc": acc, "pos_sim": pos_sim}


def supcon(
    z1: Tensor,
    z2: Tensor,
    labels: Tensor,
    temperature: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Supervised contrastive loss, in the **L_out** form.

    .. math::
        L_i = \\frac{-1}{|P(i)|} \\sum_{p \\in P(i)}
              \\log \\frac{\\exp(z_i \\cdot z_p / \\tau)}
                          {\\sum_{a \\neq i} \\exp(z_i \\cdot z_a / \\tau)}

    The normalization sits *inside* the log, averaging over positives after
    taking logs. The common mistake is ``L_in`` — taking the log of the mean of
    the positive terms — which is a different objective and empirically worse.
    The two are easy to confuse because they differ by one line and both train.

    Anchors with no positives in the batch are dropped rather than contributing
    a zero, which would silently shrink the loss in proportion to how many
    singleton classes the batch happened to contain.

    With all-distinct labels this reduces exactly to :func:`nt_xent`; a unit
    test pins that equivalence.
    """
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    y = torch.cat([labels, labels], dim=0)

    sim = z @ z.t() / temperature
    eye = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, NEG_INF)

    positives = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = positives.sum(1)
    valid = n_pos > 0
    if not bool(valid.any()):
        return torch.zeros((), device=z.device, requires_grad=True), {"n_valid_anchors": 0.0}

    # `torch.where`, not multiplication by the mask: the self-similarity
    # diagonal holds -inf, and -inf * 0 is NaN, not 0. Multiplying would poison
    # every anchor's sum -- and it would do so silently, since the loss simply
    # becomes NaN a few steps in rather than raising.
    masked_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob))
    mean_log_prob_pos = masked_log_prob.sum(1)[valid] / n_pos[valid]
    loss = -mean_log_prob_pos.mean()

    return loss, {
        "n_valid_anchors": float(valid.float().mean()),
        "mean_positives": float(n_pos[valid].float().mean()),
    }


def barlow_twins(
    z1: Tensor, z2: Tensor, lambd: float = 5e-3, eps: float = 1e-5
) -> tuple[Tensor, dict[str, float]]:
    """Barlow Twins: push the cross-correlation matrix toward the identity.

    Embeddings are standardized per dimension across the batch with **no
    learnable affine**. That is not ``nn.BatchNorm1d`` with default settings —
    a learnable scale and shift would let the network undo the standardization
    the objective depends on.
    """
    b, d = z1.shape
    z1n = (z1 - z1.mean(0)) / (z1.std(0) + eps)
    z2n = (z2 - z2.mean(0)) / (z2.std(0) + eps)

    c = (z1n.t() @ z2n) / b
    on_diag = (torch.diagonal(c) - 1).pow(2).sum()
    off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    loss = on_diag + lambd * off_diag

    return loss, {
        "bt_on_diag": float(on_diag.detach()) / d,
        "bt_off_diag": float(off_diag.detach()) / max(1, d * (d - 1)),
    }


def vicreg(
    z1: Tensor,
    z2: Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> tuple[Tensor, dict[str, float]]:
    """VICReg: variance, invariance, covariance.

    ``eps`` sits **inside** the square root of the variance term. This is not a
    numerical nicety: ``d/dv sqrt(v)`` diverges as ``v -> 0``, which is exactly
    the regime the hinge is pulling against, so omitting it produces NaNs a few
    dozen steps into training. Placing it outside the sqrt does not fix the
    gradient.
    """
    b, d = z1.shape
    inv = F.mse_loss(z1, z2)

    def variance(z: Tensor) -> Tensor:
        return F.relu(1.0 - torch.sqrt(z.var(dim=0) + eps)).mean()

    def covariance(z: Tensor) -> Tensor:
        zc = z - z.mean(0)
        cov = (zc.t() @ zc) / (b - 1)
        return (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / d

    var = variance(z1) + variance(z2)
    cov = covariance(z1) + covariance(z2)
    loss = sim_coeff * inv + std_coeff * var + cov_coeff * cov

    return loss, {
        "vicreg_inv": float(inv.detach()),
        "vicreg_var": float(var.detach()),
        "vicreg_cov": float(cov.detach()),
    }


def negative_cosine(p: Tensor, z: Tensor) -> Tensor:
    """BYOL / SimSiam similarity, ``2 - 2*cos``, in ``[0, 4]``.

    ``z`` must already be detached by the caller. Keeping the stop-gradient at
    the call site rather than hiding it here makes it visible in each method,
    which matters because it is the single line that prevents collapse.
    """
    return (2 - 2 * (F.normalize(p, dim=-1) * F.normalize(z, dim=-1)).sum(-1)).mean()


def patchify(x: Tensor, patch_size: int) -> Tensor:
    """``(B, C, L)`` -> ``(B, N, patch_size * C)``, channel-major within a patch.

    The layout must match what :class:`~iqssl.models.heads.PatchDecoder`'s output
    head is trained against; both sides go through this one function so the
    pairing cannot drift.
    """
    b, c, ell = x.shape
    if ell % patch_size:
        raise ValueError(f"length {ell} is not divisible by patch_size {patch_size}")
    n = ell // patch_size
    return x.reshape(b, c, n, patch_size).permute(0, 2, 1, 3).reshape(b, n, c * patch_size)


def unpatchify(p: Tensor, patch_size: int, in_ch: int = 2) -> Tensor:
    """Inverse of :func:`patchify`. Used by tests and reconstruction plots."""
    b, n, d = p.shape
    if d != patch_size * in_ch:
        raise ValueError(f"patch dim {d} != patch_size*in_ch = {patch_size * in_ch}")
    return p.reshape(b, n, in_ch, patch_size).permute(0, 2, 1, 3).reshape(b, in_ch, n * patch_size)


def masked_mse(
    pred: Tensor, target: Tensor, mask: Tensor, *, normalize_targets: bool = False
) -> tuple[Tensor, dict[str, float]]:
    """MSE over **masked** positions only. ``mask`` is ``(B, N)``, True = masked.

    Restricting the loss to masked positions is not an optimization: including
    the visible patches turns a prediction task into a partial autoencoder, and
    the encoder can lower the loss by copying inputs instead of inferring the
    hidden ones.

    ``normalize_targets`` standardizes each target patch to zero mean and unit
    variance (MAE's ``norm_pix_loss``). It defaults **off** here, deliberately:
    per-patch normalization erases the amplitude envelope, and for OOK the
    envelope *is* the modulation -- signal, not nuisance. Enable it only as an
    explicit ablation.
    """
    if normalize_targets:
        mean = target.mean(-1, keepdim=True)
        var = target.var(-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

    per_patch = (pred - target).pow(2).mean(-1)  # (B, N)
    denom = mask.sum().clamp_min(1)
    loss = (per_patch * mask).sum() / denom
    return loss, {"masked_frac": float(mask.float().mean())}


def masked_smooth_l1(pred: Tensor, target: Tensor, mask: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth L1 over masked positions, for the latent-prediction family.

    Smooth L1 rather than plain MSE because latent targets come from an EMA
    teacher whose scale drifts over training; the linear tail keeps an early
    large-residual batch from dominating the step.
    """
    per_tok = F.smooth_l1_loss(pred, target, beta=beta, reduction="none").mean(-1)  # (B, N)
    return (per_tok * mask).sum() / mask.sum().clamp_min(1)
