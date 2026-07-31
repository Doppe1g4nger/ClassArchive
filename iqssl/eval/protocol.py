"""Orchestrate the protocol over one run, and write ``eval.json``.

The grid is {linear probe, kNN, finetune} x {1%, 10%, 100% labels} x
{emitter, modulation}, plus the nuisance-regression probe and accuracy by SNR
quartile. Label subsets come from the lists stored at dataset build time
(:meth:`IQDataset.label_subset`), so two methods finetuning "on 1% of labels"
are finetuning on *the same samples* — otherwise part of the gap between them
would be which labels each happened to draw.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from iqssl.data.dataset import IQDataset
from iqssl.data.params import NUISANCE_FIELDS
from iqssl.eval import probes
from iqssl.eval.features import POOL, SplitFeatures, extract
from iqssl.eval.loading import check_dataset_hash, load_encoder, load_run_config
from iqssl.utils.logging_ import get_logger, write_json

log = get_logger(__name__)

LABEL_FRACTIONS = (0.01, 0.1, 1.0)
LABEL_AXES = ("emitter", "modulation")
SNR_QUARTILES = 4


def evaluate_run(
    run_dir: str | Path,
    data_root: str | Path | None = None,
    *,
    split_variant: str = "iid",
    device: str = "cpu",
    allow_hash_mismatch: bool = False,
    skip_finetune: bool = False,
) -> dict:
    """Evaluate one pretrained run. Returns the report and writes ``eval.json``.

    ``skip_finetune`` exists for smoke iterations only; a reported result
    without the finetune column is not the protocol, and the written report
    says so explicitly rather than leaving an empty dict to be misread.
    """
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    root = Path(data_root) if data_root is not None else Path(cfg.data.root)

    dataset_hash = check_dataset_hash(run_dir, root, allow_mismatch=allow_hash_mismatch)

    # One dataset object per label axis: they expose the same rows in the same
    # order (the split is axis-independent), differing only in which stored
    # label subset label_subset() consults.
    train_by_axis = {
        axis: IQDataset(
            root, "train", split_variant=split_variant, primary_label=axis, random_crop=False
        )
        for axis in LABEL_AXES
    }
    test_ds = IQDataset(
        root, "test", split_variant=split_variant, primary_label="emitter", random_crop=False
    )
    train_ds = train_by_axis["emitter"]

    encoder = load_encoder(run_dir, train_ds.crop_len, train_ds.num_primary_classes)

    log.info("extracting frozen features (%d train / %d test)", len(train_ds), len(test_ds))
    tr = extract(encoder, train_ds, device)
    te = extract(encoder, test_ds, device)

    # Raw test buffers, loaded once and shared by every finetune call. Crops
    # are deterministic (random_crop=False), so this is the same tensor the
    # feature extraction saw.
    xte = None
    if not skip_finetune:
        xte = torch.stack([test_ds[i]["x"] for i in range(len(test_ds))])

    report: dict = {
        "run": str(run_dir),
        "method": str(cfg.method.name),
        "dataset_hash": dataset_hash,
        "split_variant": split_variant,
        "pool": POOL,
        "label_fractions": list(LABEL_FRACTIONS),
        "finetune_included": not skip_finetune,
        "axes": {
            axis: _evaluate_axis(axis, train_by_axis[axis], tr, te, xte, encoder, device)
            for axis in LABEL_AXES
        },
        "nuisance_r2": dict(
            zip(
                NUISANCE_FIELDS,
                probes.nuisance_r2(tr.z, tr.nuisance, te.z, te.nuisance),
                strict=True,
            )
        ),
    }

    write_json(run_dir / "eval.json", report)
    log.info("wrote %s", run_dir / "eval.json")
    return report


def _evaluate_axis(
    axis: str,
    train_ds: IQDataset,
    tr: SplitFeatures,
    te: SplitFeatures,
    xte: Tensor | None,
    encoder: nn.Module,
    device: str,
) -> dict:
    ytr_full, yte = tr.labels(axis), te.labels(axis)
    n_classes = int(max(ytr_full.max(), yte.max())) + 1

    out: dict = {"linear_probe": {}, "knn": {}, "finetune": {}}
    full_label_predictions: np.ndarray | None = None

    for frac in LABEL_FRACTIONS:
        rows = (
            np.arange(len(ytr_full))
            if frac >= 1.0
            else np.asarray(train_ds.label_subset(frac), dtype=np.int64)
        )
        ztr, ytr = tr.z[rows], ytr_full[rows]
        key = f"{frac:g}"
        log.info("[%s @ %s] %d labelled samples", axis, key, len(rows))

        acc, predictions = probes.linear_probe(ztr, ytr, te.z, yte, n_classes, device=device)
        out["linear_probe"][key] = acc
        if frac >= 1.0:
            full_label_predictions = predictions

        out["knn"][key] = probes.knn_probe(ztr, ytr, te.z, yte, n_classes)

        if xte is not None:
            ft_rows = rows
            if len(ft_rows) > probes.FINETUNE_CAP:
                # The cap draw is seeded and identical for every method, so it
                # cannot favour anyone; sorting keeps read order disk-friendly.
                rng = np.random.default_rng(probes.PROBE_SEED)
                ft_rows = np.sort(rng.permutation(ft_rows)[: probes.FINETUNE_CAP])
            xtr = torch.stack([train_ds[int(i)]["x"] for i in ft_rows])
            out["finetune"][key] = probes.finetune_probe(
                encoder, xtr, ytr_full[ft_rows], xte, yte, n_classes, pool=POOL, device=device
            )

    assert full_label_predictions is not None
    out["snr_quartiles"] = _snr_quartiles(te.snr, full_label_predictions, yte)
    return out


def _snr_quartiles(snr: np.ndarray, predictions: np.ndarray, y: np.ndarray) -> list[dict]:
    """Accuracy by SNR quartile, sliced from the full-label probe's predictions.

    One probe evaluated per bin, never one probe per bin: retraining on a bin
    would confound the bin's difficulty with its training-set size.
    """
    edges = np.quantile(snr, np.linspace(0, 1, SNR_QUARTILES + 1))
    rows = []
    for lo, hi in itertools.pairwise(edges):
        m = (snr >= lo) & (snr <= hi)
        rows.append(
            {
                "snr_lo": float(lo),
                "snr_hi": float(hi),
                "n": int(m.sum()),
                "accuracy": float((predictions[m] == y[m]).mean()) if m.any() else float("nan"),
            }
        )
    return rows
