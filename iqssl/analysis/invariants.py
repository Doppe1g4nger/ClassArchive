"""What must agree between two runs before their numbers may be averaged.

This is :func:`iqssl.eval.loading.check_dataset_hash` generalized from one field
to the whole fairness contract. The dataset hash was the obvious one; it is not
the only way a table can silently pool incomparable runs. An encoder width that
changed between sweeps, an augmentation policy someone overrode for one method,
a schedule shortened mid-experiment — each produces a results table that reads
perfectly and means nothing.

The split between the two lists below *is* the fairness contract, so the lists
are the place to argue about it rather than a prose paragraph that code does not
read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MUST_AGREE: tuple[str, ...] = (
    "dataset_hash",
    "encoder",
    "epochs",
    "batch_size",
    "warmup_frac",
    "grad_clip",
    "augment",
    "crop_len",
    "split_variant",
)
"""Held constant by the contract. Disagreement means the runs are measuring
different experiments and averaging them produces a number about neither."""

MAY_DIFFER: tuple[str, ...] = (
    "method",
    "seed",
    "optimizer",
    "base_lr",
    "weight_decay",
)
"""Deliberately varied. ``method`` and ``seed`` are the axes being compared;
optimizer family and the tuned knobs vary because forcing one family on every
method would measure optimizer tolerance rather than objective quality."""


class IncomparableRuns(RuntimeError):
    """Raised when runs disagree on something the contract holds constant."""


@dataclass(frozen=True)
class Violation:
    key: str
    values: tuple[Any, ...]
    runs: tuple[str, ...]

    def describe(self) -> str:
        pairs = ", ".join(f"{r}={v!r}" for r, v in zip(self.runs, self.values, strict=True))
        return f"{self.key}: {pairs}"


def check_invariants(rows: list[dict], *, raise_on_violation: bool = True) -> list[Violation]:
    """Verify every run in ``rows`` agrees on the held-constant fields.

    Returns the violations found. Raising is the default because the failure
    this guards against is silent by construction: nothing downstream of a bad
    pool can detect that it happened, and a warning in a log is not read by the
    person reading the table six months later.

    A field missing from *every* run is not a violation — older runs predate
    some keys — but a field present in some and absent in others is, since that
    is exactly the case where a default gets silently substituted.
    """
    if len(rows) < 2:
        return []

    violations: list[Violation] = []
    for key in MUST_AGREE:
        present = [(r.get("run", "?"), r[key]) for r in rows if key in r]
        if not present:
            continue
        if len(present) != len(rows):
            missing = [r.get("run", "?") for r in rows if key not in r]
            violations.append(
                Violation(
                    key=key, values=("<present>", "<absent>"), runs=(present[0][0], missing[0])
                )
            )
            continue

        distinct = {_hashable(v) for _, v in present}
        if len(distinct) > 1:
            first = present[0]
            other = next(p for p in present if _hashable(p[1]) != _hashable(first[1]))
            violations.append(
                Violation(key=key, values=(first[1], other[1]), runs=(first[0], other[0]))
            )

    if violations and raise_on_violation:
        raise IncomparableRuns(
            "refusing to pool runs that disagree on fields the fairness contract "
            "holds constant:\n  "
            + "\n  ".join(v.describe() for v in violations)
            + "\n\nAveraging across these produces a number about neither experiment. "
            "Aggregate them separately, or pass --allow-incomparable if you are "
            "deliberately inspecting a mixture and know the table is not a result."
        )
    return violations


def _hashable(value: Any) -> Any:
    """Collapse nested config fragments to something comparable by value.

    ``encoder`` is a dict; comparing dicts by identity would let two structurally
    identical encoders read as different, and comparing by ``==`` cannot go in a
    set. Sorted items give a stable, order-insensitive key.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value
