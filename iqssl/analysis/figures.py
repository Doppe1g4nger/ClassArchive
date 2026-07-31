"""Figures: the compute-vs-accuracy Pareto front, and accuracy vs SNR.

The Pareto plot is the fairness contract's own admission that equal epochs is
not equal compute. MAE's encoder sees a quarter of the tokens; BYOL runs a
teacher pass it never backprops through. Ranking those on accuracy alone would
reward whichever objective happened to be given the most FLOPs per epoch, so
the cost axis is reported beside the score.

It is honest per method only because ``Method.encoder_passes_per_step`` makes
each one declare its own spend — a loop-side estimate would have to know which
forwards were teacher passes, which means branching on method type.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI or on a headless box
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COST_FIELD = "compute/tokens_seen"
"""Tokens, not wall-clock. Wall-clock measures the machine that happened to be
free; tokens measure the objective. Both are recorded, and the axis label says
which is plotted."""


def pareto_front(cost: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Indices on the frontier: no other point is both cheaper and better.

    Ties count as dominated only if strictly beaten on one axis and matched on
    the other, so two identical points both stay on the front rather than one
    arbitrarily displacing the other.
    """
    order = np.argsort(cost)
    front, best = [], -np.inf
    for i in order:
        if score[i] > best:
            front.append(i)
            best = score[i]
    return np.array(front, dtype=int)


def compute_vs_accuracy(
    scores: pd.DataFrame,
    metas: list[dict],
    out_path: str | Path,
    *,
    axis: str = "emitter",
    probe: str = "linear_probe",
    fraction: float = 1.0,
) -> Path:
    cost_by_run = {m["run"]: m.get(COST_FIELD) for m in metas}
    subset = scores[
        (scores["axis"] == axis) & (scores["probe"] == probe) & (scores["fraction"] == fraction)
    ].copy()
    subset["cost"] = subset["run"].map(cost_by_run)
    subset = subset.dropna(subset=["cost"])
    if subset.empty:
        raise ValueError(
            f"no runs carry {COST_FIELD}; the Pareto plot needs summary.json from "
            "training, so evaluate runs that were produced by iqssl-pretrain."
        )

    grouped = subset.groupby("method").agg(
        cost=("cost", "mean"), score=("score", "mean"), spread=("score", "std")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        grouped["cost"],
        grouped["score"],
        yerr=grouped["spread"].fillna(0.0),
        fmt="o",
        capsize=3,
        color="#333333",
    )
    for method, row in grouped.iterrows():
        ax.annotate(
            str(method), (row["cost"], row["score"]), textcoords="offset points", xytext=(6, 4)
        )

    front = pareto_front(grouped["cost"].to_numpy(), grouped["score"].to_numpy())
    if len(front):
        ax.plot(
            grouped["cost"].to_numpy()[front],
            grouped["score"].to_numpy()[front],
            "--",
            color="#c0392b",
            label="Pareto front",
        )
        ax.legend()

    ax.set_xscale("log")
    ax.set_xlabel("tokens seen by the encoder (log)")
    ax.set_ylabel(f"{probe} accuracy, {axis} @ {fraction:g} labels")
    ax.set_title("Compute vs accuracy")
    ax.grid(alpha=0.3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def accuracy_vs_snr(scores: pd.DataFrame, out_path: str | Path, *, axis: str = "emitter") -> Path:
    """Per-method accuracy across the stored SNR quartiles."""
    subset = scores[(scores["axis"] == axis) & (scores["probe"] == "snr_band")]
    if subset.empty:
        raise ValueError("no snr_band rows; evaluate runs with the full protocol first")

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in subset.groupby("method"):
        band = group.groupby("fraction")["score"].mean().sort_index()
        ax.plot(band.index, band.to_numpy(), marker="o", label=str(method))

    ax.set_xlabel("SNR quartile lower edge (dB)")
    ax.set_ylabel(f"{axis} accuracy")
    ax.set_title("Accuracy vs SNR")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
