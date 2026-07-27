"""Train/val/test partitions, and the shared few-label subsets.

Four split variants, because "does this representation generalize?" is four
different questions and an IID split answers only the easiest one:

``iid``
    Random. Measures in-distribution quality.
``holdout_emitters``
    Whole devices unseen in training. Measures whether the representation learned
    *what a fingerprint is* rather than memorizing sixteen of them.
``holdout_snr``
    The low-SNR tail reserved for test. Measures robustness where it is scarce.
``holdout_channel``
    The richest multipath reserved for test. Multipath is the dominant destroyer
    of the fingerprint, so this is the hardest of the four.

The label subsets are shared across every method by construction. If SimCLR and
BYOL each drew their own 1% of labels to finetune on, part of the gap between
them would be which labels they happened to get.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_VARIANTS = ("iid", "holdout_emitters", "holdout_snr", "holdout_channel")

DEFAULT_FRACTIONS = (0.7, 0.1, 0.2)
"""train / val / test."""

LABEL_AXES = ("emitter_id", "modulation")


def make_splits(
    meta: pd.DataFrame,
    variant: str = "iid",
    seed: int = 0,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> np.ndarray:
    """Assign each row to ``train``/``val``/``test``. Returns an array of str."""
    if variant not in SPLIT_VARIANTS:
        raise ValueError(f"unknown split variant {variant!r}; options: {list(SPLIT_VARIANTS)}")

    if variant == "iid":
        return _iid(meta, seed, fractions)
    if variant == "holdout_emitters":
        return _holdout_emitters(meta, seed, fractions)
    if variant == "holdout_snr":
        return _holdout_snr(meta, seed, fractions)
    return _holdout_channel(meta, seed, fractions)


def _blank(n: int) -> np.ndarray:
    return np.full(n, "train", dtype=object)


def _iid(meta: pd.DataFrame, seed: int, fr: tuple[float, float, float]) -> np.ndarray:
    """Stratified by emitter *and* modulation.

    A plain shuffle can leave a rare class out of training entirely, at which
    point the probe reports a number about the draw rather than about the method.
    """
    rng = np.random.default_rng(seed)
    out = _blank(len(meta))
    strata = meta["emitter_id"].astype(str) + "|" + meta["modulation"].astype(str)

    for _, rows in meta.groupby(strata.to_numpy(), sort=True).groups.items():
        idx = rng.permutation(np.asarray(rows))
        n = len(idx)
        n_val = round(n * fr[1])
        n_test = round(n * fr[2])
        out[idx[:n_test]] = "test"
        out[idx[n_test : n_test + n_val]] = "val"
    return out


def _holdout_emitters(meta: pd.DataFrame, seed: int, fr: tuple[float, float, float]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = np.unique(meta["emitter_id"].to_numpy())
    if len(ids) < 3:
        raise ValueError(
            f"holdout_emitters needs at least 3 emitters to form three disjoint "
            f"splits, the dataset has {len(ids)}"
        )
    shuffled = rng.permutation(ids)
    n_test = max(1, round(len(ids) * fr[2]))
    n_val = max(1, round(len(ids) * fr[1]))
    test_ids = set(shuffled[:n_test].tolist())
    val_ids = set(shuffled[n_test : n_test + n_val].tolist())

    out = _blank(len(meta))
    e = meta["emitter_id"].to_numpy()
    out[np.isin(e, list(test_ids))] = "test"
    out[np.isin(e, list(val_ids))] = "val"
    return out


def _holdout_snr(meta: pd.DataFrame, seed: int, fr: tuple[float, float, float]) -> np.ndarray:
    """Reserve the lowest-SNR tail for test, the next band for val.

    Sorted rather than thresholded so the split sizes are exact regardless of how
    the preset's SNR prior is shaped.
    """
    snr = meta["snr_db_realized"].to_numpy()
    order = np.argsort(snr, kind="stable")
    n = len(order)
    n_test = round(n * fr[2])
    n_val = round(n * fr[1])

    out = _blank(n)
    out[order[:n_test]] = "test"
    out[order[n_test : n_test + n_val]] = "val"
    return out


def _holdout_channel(meta: pd.DataFrame, seed: int, fr: tuple[float, float, float]) -> np.ndarray:
    """Reserve the richest multipath for test.

    The guard is not defensive boilerplate. With a single-tap prior every buffer
    ties for "richest", so a naive threshold sends the *whole dataset* to test and
    silently produces an empty training split -- a dataset that builds fine, loads
    fine, and trains on nothing.
    """
    taps = meta["n_taps"].to_numpy()
    distinct = np.unique(taps)
    if len(distinct) < 2:
        raise ValueError(
            f"holdout_channel needs more than one tap count to hold any out, but "
            f"every buffer has n_taps={distinct[0]}. This preset has no multipath "
            f"diversity; use a preset whose n_taps_choices has several entries."
        )

    # Whole tap counts move together, richest first: test, then val, then
    # whatever remains is train. Splitting *within* a tap count would put
    # statistically identical channels on both sides and defeat the variant.
    # The last tap count is never consumed, so training is never empty -- with
    # only two distinct counts that means val is empty, which is the honest
    # outcome rather than a fabricated one.
    out = _blank(len(meta))
    descending = sorted(distinct.tolist(), reverse=True)
    budgets = {"test": len(meta) * fr[2], "val": len(meta) * fr[1]}

    remaining = list(descending)
    for part in ("test", "val"):
        taken = 0
        while len(remaining) > 1 and taken < budgets[part]:
            t = remaining.pop(0)
            mask = taps == t
            out[mask] = part
            taken += int(mask.sum())
    return out


def make_label_subsets(
    meta: pd.DataFrame,
    split: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.1),
    seed: int = 0,
) -> dict[str, dict[str, list[int]]]:
    """The shared few-label subsets, keyed ``[label_axis][str(fraction)]``.

    Two properties the finetuning protocol depends on:

    **Nested.** The 1% subset is a subset of the 10% one. Otherwise the
    label-efficiency curve partly measures which samples each fraction drew.

    **Class-covering.** Every class appears at least once at every fraction. A
    plain random 1% can miss classes entirely, and a method's few-label score
    would then depend on the draw rather than on the representation.
    """
    train_idx = np.flatnonzero(split == "train")
    if len(train_idx) == 0:
        raise ValueError("the training split is empty; label subsets would be meaningless")

    out: dict[str, dict[str, list[int]]] = {}
    for axis in LABEL_AXES:
        rng = np.random.default_rng(seed)
        labels = meta[axis].to_numpy()[train_idx]
        classes = np.unique(labels)

        # One shared shuffled order per class, sliced deeper as the fraction
        # grows. Nesting falls out of that rather than being enforced afterwards.
        per_class = {c: rng.permutation(train_idx[labels == c]) for c in classes}

        axis_out: dict[str, list[int]] = {}
        for frac in sorted(fractions):
            take: list[int] = []
            for c in classes:
                pool = per_class[c]
                # At least one per class, hence the max(1, ...).
                k = max(1, round(len(pool) * frac))
                take.extend(int(i) for i in pool[:k])
            axis_out[str(frac)] = sorted(take)
        out[axis] = axis_out
    return out
