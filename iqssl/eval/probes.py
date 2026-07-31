"""The probes, with their budgets fixed as module constants.

Every number below applies to every method identically. They are constants, not
parameters, because the protocol *is* the fairness contract here: the moment a
probe budget becomes configurable, the first "quick experiment" that raises one
method's probe epochs invalidates the table it lands in. Changing a budget is
allowed — as a reviewed edit to this file, which changes it for everyone at
once.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from iqssl.models.heads import LinearClassifier
from iqssl.utils.seed import seed_everything

PROBE_EPOCHS = 40
PROBE_LR = 1e-3
PROBE_BATCH = 256
PROBE_WEIGHT_DECAY = 0.0

KNN_K = 20
KNN_TEMPERATURE = 0.07

FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4
FINETUNE_BATCH = 64
FINETUNE_CAP = 20_000
"""Finetuning at 100% labels on a 200k-sample dataset would dwarf pretraining
itself; every method finetunes on at most this many samples, drawn with the
same seed, so the cap cannot favour anyone."""

RIDGE_LAMBDA = 1e-2
PROBE_SEED = 0


def _accuracy(logits: Tensor, y: Tensor) -> float:
    return float((logits.argmax(-1) == y).float().mean())


def linear_probe(
    ztr: Tensor, ytr: np.ndarray, zte: Tensor, yte: np.ndarray, n_classes: int, device: str = "cpu"
) -> tuple[float, np.ndarray]:
    """Frozen-feature linear probe. Returns ``(accuracy, test_predictions)``.

    Uses :class:`LinearClassifier`, whose affine-free BatchNorm is MAE's own
    probe protocol — masked-method features have very different per-dimension
    scales than contrastive ones, and without the BN the *protocol* would decide
    the ranking.

    Predictions come back with the score so per-slice reporting (accuracy by
    SNR bin) can reuse the same trained probe: retraining per slice would
    confound slice difficulty with training-set composition.
    """
    seed_everything(PROBE_SEED)
    dev = torch.device(device)
    head = LinearClassifier(ztr.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)

    ztr_d, ytr_d = ztr.to(dev), torch.from_numpy(ytr).to(dev)
    g = torch.Generator().manual_seed(PROBE_SEED)
    head.train()
    for _ in range(PROBE_EPOCHS):
        perm = torch.randperm(len(ztr_d), generator=g)
        for s in range(0, len(perm), PROBE_BATCH):
            idx = perm[s : s + PROBE_BATCH]
            if len(idx) < 2:
                continue  # BatchNorm needs at least two rows
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(ztr_d[idx]), ytr_d[idx]).backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        pred = head(zte.to(dev)).argmax(-1).cpu()
    return float((pred == torch.from_numpy(yte)).float().mean()), pred.numpy()


def knn_probe(ztr: Tensor, ytr: np.ndarray, zte: Tensor, yte: np.ndarray, n_classes: int) -> float:
    """Cosine-weighted kNN, the standard SSL protocol (k=20, T=0.07).

    Hyperparameter-free in the way that matters: no training, so nothing to
    tune, so nothing to tune unevenly. This is the probe least able to flatter
    anyone, which is why it is reported alongside the trained ones.
    """
    a = F.normalize(ztr.float(), dim=1)
    b = F.normalize(zte.float(), dim=1)
    y = torch.from_numpy(ytr)

    k = min(KNN_K, len(a))
    hits = 0
    for s in range(0, len(b), 512):
        sim = b[s : s + 512] @ a.T  # (b, N_train)
        topv, topi = sim.topk(k, dim=1)
        weights = (topv / KNN_TEMPERATURE).exp()
        votes = torch.zeros(len(topi), n_classes)
        votes.scatter_add_(1, y[topi], weights)
        hits += int((votes.argmax(1) == torch.from_numpy(yte[s : s + 512])).sum())
    return hits / len(b)


def finetune_probe(
    encoder: nn.Module,
    xtr: Tensor,
    ytr: np.ndarray,
    xte: Tensor,
    yte: np.ndarray,
    n_classes: int,
    pool: str = "mean",
    device: str = "cpu",
) -> float:
    """End-to-end finetune from the pretrained weights.

    The encoder is deep-copied and unfrozen, so the caller's frozen copy — which
    the other probes rely on — is untouched, and repeated calls (per label
    fraction) all start from the same pretrained point rather than from the
    previous fraction's finetuned one.
    """
    seed_everything(PROBE_SEED)
    dev = torch.device(device)

    model = copy.deepcopy(encoder).to(dev)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    head = LinearClassifier(int(model.embed_dim), n_classes, use_bn=False).to(dev)  # type: ignore[arg-type]

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=FINETUNE_LR, weight_decay=0.05)

    ytr_d = torch.from_numpy(ytr).to(dev)
    g = torch.Generator().manual_seed(PROBE_SEED)
    for _ in range(FINETUNE_EPOCHS):
        perm = torch.randperm(len(xtr), generator=g)
        for s in range(0, len(perm) - 1, FINETUNE_BATCH):
            idx = perm[s : s + FINETUNE_BATCH]
            opt.zero_grad(set_to_none=True)
            h = model(xtr[idx].to(dev)).pooled(pool)
            F.cross_entropy(head(h), ytr_d[idx]).backward()
            opt.step()

    model.eval()
    head.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(xte), 256):
            preds.append(head(model(xte[s : s + 256].to(dev)).pooled(pool)).argmax(-1).cpu())
    return float((torch.cat(preds) == torch.from_numpy(yte)).float().mean())


def nuisance_r2(ztr: Tensor, ntr: np.ndarray, zte: Tensor, nte: np.ndarray) -> list[float]:
    """Closed-form ridge regression, features -> standardized nuisance vector.

    The "what did the representation discard?" measurement. Per-field R² on the
    test split: near 1 means the nuisance is still linearly readable, near 0 (or
    below — worse than predicting the mean) means it was discarded. Closed form
    because a probe with an optimizer would add budget questions to a question
    that has an exact answer.
    """
    x = ztr.double().numpy()
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)  # bias column
    a = x.T @ x + RIDGE_LAMBDA * np.eye(x.shape[1])
    w = np.linalg.solve(a, x.T @ ntr.astype(np.float64))

    xt = zte.double().numpy()
    xt = np.concatenate([xt, np.ones((len(xt), 1))], axis=1)
    pred = xt @ w

    resid = ((nte - pred) ** 2).sum(0)
    total = ((nte - nte.mean(0)) ** 2).sum(0)
    return [
        float(1.0 - r / t) if t > 0 else float("nan") for r, t in zip(resid, total, strict=True)
    ]
