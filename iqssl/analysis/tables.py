"""Thesis tables: mean and spread over seeds, method x policy, nuisance R².

Every table reports a spread alongside a mean, and a single-seed cell reports
**no** spread rather than zero. A zero error bar on one seed is a claim of
perfect reproducibility that nobody made, and it is the sort of thing a reader
takes at face value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROBE_ORDER = ("linear_probe", "knn", "finetune")
PROBE_LABELS = {"linear_probe": "linear", "knn": "kNN", "finetune": "finetune"}


def _agg(group: pd.DataFrame) -> pd.Series:
    scores = group["score"].to_numpy(dtype=float)
    return pd.Series(
        {
            "mean": float(np.mean(scores)),
            # ddof=1: the sample standard deviation of a handful of seeds, not
            # the population one. Undefined for a single seed, which is the
            # honest answer -- pandas returns NaN and the formatter prints "--".
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"),
            "n_seeds": len(scores),
        }
    )


def headline_table(scores: pd.DataFrame, axis: str = "emitter") -> pd.DataFrame:
    """method x probe x label-fraction, aggregated over seeds."""
    subset = scores[(scores["axis"] == axis) & (scores["probe"].isin(PROBE_ORDER))]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["method", "probe", "fraction"], sort=True)[["score"]]
        .apply(_agg, include_groups=False)
        .reset_index()
    )


def policy_interaction_table(
    scores: pd.DataFrame, metas: list[dict], axis: str = "emitter", fraction: float = 1.0
) -> pd.DataFrame:
    """method x augmentation policy — the interaction the thesis reports.

    Reported as an interaction rather than a per-method ranking on purpose: a
    contrastive method becomes invariant to whatever it is augmented with, so
    "method A beats method B" is only meaningful alongside the policy both ran
    under. Comparing a method against another method plus a hand-designed prior
    is the confound this table exists to expose.
    """
    policy = {m["run"]: m.get("augment") for m in metas}
    subset = scores[
        (scores["axis"] == axis)
        & (scores["probe"] == "linear_probe")
        & (scores["fraction"] == fraction)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["policy"] = subset["run"].map(policy)
    return (
        subset.groupby(["method", "policy"], sort=True)[["score"]]
        .apply(_agg, include_groups=False)
        .reset_index()
    )


def nuisance_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-method, per-field R² — what each representation retained.

    Near 1 means the nuisance is still linearly readable off the features; near
    0 means it was discarded. Neither is "better" on its own, which is why this
    is a table and not a score: discarding CFO is success, discarding the
    emitter fingerprint is failure, and only the reader knows which column is
    which.
    """
    subset = scores[scores["axis"] == "nuisance"]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["method", "probe"], sort=True)[["score"]]
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"probe": "nuisance_field"})
    )


def to_markdown(table: pd.DataFrame, title: str) -> str:
    """Render with `mean ± std`, or a bare mean where there is only one seed."""
    if table.empty:
        return f"### {title}\n\n_(no runs)_\n"

    shown = table.copy()
    shown["value"] = [
        f"{m:.3f}" if np.isnan(s) else f"{m:.3f} ± {s:.3f}"
        for m, s in zip(shown["mean"], shown["std"], strict=True)
    ]
    keys = [c for c in shown.columns if c not in ("mean", "std", "n_seeds", "value")]
    lines = [f"### {title}", ""]

    seeds = sorted(set(shown["n_seeds"]))
    if seeds == [1]:
        lines += ["_Single seed: no spread reported. These are not results._", ""]

    lines.append("| " + " | ".join([*keys, "score"]) + " |")
    lines.append("| " + " | ".join(["---"] * (len(keys) + 1)) + " |")
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join([str(row[k]) for k in keys] + [row["value"]]) + " |")
    return "\n".join(lines) + "\n"
