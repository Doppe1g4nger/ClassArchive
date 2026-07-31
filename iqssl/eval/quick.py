"""The validation probe that hyperparameter search selects on.

Separate from :mod:`iqssl.eval.protocol` because it answers a different
question. The protocol reports what a representation is worth, on the test
split, once. This runs during tuning, on the **validation** split, many times,
and its only job is to rank configurations of one method against each other.

Why a probe at all, rather than the pretraining loss Hydra would otherwise
optimize: **for BYOL and SimSiam a collapsed representation has near-zero
loss.** Tuning on the objective value would not merely be uninformative, it
would reliably select the collapsed configuration — the one failure mode those
methods are designed around. Across methods the losses are not even on the same
scale (NT-Xent ~2, Barlow ~10⁴), so nothing about a loss number tracks
representation quality.

The probe reuses :func:`iqssl.eval.features.extract` and
:func:`iqssl.eval.probes.linear_probe` unchanged, so the quantity being tuned
for is the same quantity that gets reported. A cheaper bespoke probe here would
mean selecting configurations on one measurement and publishing another.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import nn

from iqssl.data.dataset import IQDataset
from iqssl.eval.features import extract
from iqssl.eval.probes import knn_probe, linear_probe
from iqssl.methods.base import rankme
from iqssl.utils.logging_ import get_logger

log = get_logger(__name__)

VAL_PROBE_CAP = 8_000
"""Cap on validation buffers used per trial. Tuning runs this many times; the
full protocol is not capped. Identical for every method and every trial, so it
cannot tilt a comparison."""


def val_probe_score(
    encoder: nn.Module,
    data_root: str | Path,
    *,
    split_variant: str = "iid",
    primary_label: str = "emitter",
    device: str = "cpu",
    crop_len: int | None = None,
) -> dict[str, float]:
    """Fit a linear probe on ``train`` and score it on ``val``.

    Returns the probe accuracy plus collapse diagnostics. **The test split is
    never opened here**, and that is the point rather than an implementation
    detail: nine trials across twelve methods selecting on test would leak it
    into every headline number in the thesis, invisibly and irreversibly.
    ``tests/test_hpo.py`` asserts the omission.
    """
    train = IQDataset(
        data_root,
        "train",
        split_variant=split_variant,
        primary_label=primary_label,
        random_crop=False,
        crop_len=crop_len,
    )
    val = IQDataset(
        data_root,
        "val",
        split_variant=split_variant,
        primary_label=primary_label,
        random_crop=False,
        crop_len=crop_len,
    )

    tr = extract(encoder, train, device, limit=VAL_PROBE_CAP)
    va = extract(encoder, val, device, limit=VAL_PROBE_CAP)

    n_classes = train.num_primary_classes
    ytr, yva = tr.labels(primary_label), va.labels(primary_label)

    accuracy, _ = linear_probe(tr.z, ytr, va.z, yva, n_classes, device=device)
    knn = knn_probe(tr.z, ytr, va.z, yva, n_classes)

    # Reported alongside so a trial that scores well *because* it collapsed is
    # legible in the sweep log rather than silently winning.
    z = tr.z.float()
    return {
        "val_probe_acc": float(accuracy),
        "val_knn_acc": float(knn),
        "val_rankme": float(rankme(z)),
        "val_std_min": float(z.std(0).min()),
        "chance": 1.0 / n_classes,
    }


def objective_from(scores: dict[str, float]) -> float:
    """The scalar Optuna maximizes. Kept in one place so the sweep and the log
    can never disagree about which number was being optimized."""
    return float(scores["val_probe_acc"])


def format_scores(scores: dict[str, float]) -> str:
    return (
        f"val_probe {scores['val_probe_acc']:.4f} (chance {scores['chance']:.3f})  "
        f"val_knn {scores['val_knn_acc']:.4f}  "
        f"rankme {scores['val_rankme']:.1f}  std_min {scores['val_std_min']:.2e}"
    )


def is_collapsed(scores: dict[str, float], min_rankme: float = 2.0) -> bool:
    """Cheap collapse flag for the sweep log, not a hard failure.

    A collapsed trial should lose on its probe score anyway; this only makes the
    reason visible. Erroring instead would be wrong — a method *can* legitimately
    have low effective rank early in a short tuning schedule.
    """
    return bool(np.isnan(scores["val_rankme"]) or scores["val_rankme"] < min_rankme)
