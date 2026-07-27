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
