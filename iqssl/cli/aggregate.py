"""Pool evaluated runs into the thesis tables and figures.

    iqssl-aggregate --root outputs/main_comparison --out results/main
    iqssl-aggregate --root outputs/smoke_cpu --out results/smoke --allow-incomparable

argparse, like ``iqssl-evaluate`` and for the same reason: the pooling rules are
the fairness contract, not configuration. The one thing this command must never
be talked into is averaging runs that were never comparable — a table computed
over two dataset versions or two augmentation policies reads exactly like a
correct one, and nothing downstream can tell.

``--allow-incomparable`` exists because inspecting a deliberate mixture is a
real need. It prints what it is overriding, and the written report records that
it was used, so a table produced that way cannot later be mistaken for a result.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from iqssl.analysis.collect import collect_runs
from iqssl.analysis.figures import accuracy_vs_snr, compute_vs_accuracy
from iqssl.analysis.invariants import check_invariants
from iqssl.analysis.tables import (
    headline_table,
    nuisance_table,
    policy_interaction_table,
    to_markdown,
)
from iqssl.utils.logging_ import get_logger, setup_console_logging, write_json

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", required=True, type=Path, help="directory of pretraining runs")
    p.add_argument("--out", required=True, type=Path, help="where tables and figures go")
    p.add_argument("--axis", default="emitter", choices=["emitter", "modulation"])
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--allow-incomparable",
        action="store_true",
        help="pool runs that disagree on a held-constant field; the output records that "
        "this was used, because such a table is not a result",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging(logging.INFO)

    scores, metas = collect_runs(args.root)
    if scores.empty:
        log.error("no eval.json found under %s; run iqssl-evaluate first", args.root)
        return 1
    log.info("collected %d runs (%d score rows)", len(metas), len(scores))

    violations = check_invariants(metas, raise_on_violation=not args.allow_incomparable)
    if violations:
        log.warning("OVERRIDDEN -- pooling runs that disagree on:")
        for v in violations:
            log.warning("  %s", v.describe())

    args.out.mkdir(parents=True, exist_ok=True)
    head = headline_table(scores, axis=args.axis)
    policy = policy_interaction_table(scores, metas, axis=args.axis)
    nuisance = nuisance_table(scores)

    for name, table in (("headline", head), ("policy", policy), ("nuisance", nuisance)):
        if not table.empty:
            table.to_csv(args.out / f"{name}.csv", index=False)

    report = "\n".join(
        [
            f"# IQSSL results -- {args.root}",
            "",
            f"{len(metas)} runs, axis = {args.axis}.",
            "",
            (
                "> **Pooled across incomparable runs (--allow-incomparable). "
                "This table is not a result.**\n"
                if violations
                else ""
            ),
            to_markdown(head, f"Headline: {args.axis}"),
            to_markdown(policy, "Method x augmentation policy"),
            to_markdown(nuisance, "Nuisance R^2 (near 0 = discarded)"),
        ]
    )
    (args.out / "results.md").write_text(report)

    write_json(
        args.out / "provenance.json",
        {
            "root": str(args.root),
            "n_runs": len(metas),
            "axis": args.axis,
            "incomparable_override": bool(violations),
            "violations": [v.describe() for v in violations],
            "runs": [m.get("run") for m in metas],
        },
    )

    if not args.no_figures:
        for label, fn, path in (
            ("compute-vs-accuracy", compute_vs_accuracy, args.out / "pareto.png"),
            ("accuracy-vs-SNR", accuracy_vs_snr, args.out / "snr.png"),
        ):
            try:
                written = (
                    fn(scores, metas, path, axis=args.axis)  # type: ignore[operator]
                    if fn is compute_vs_accuracy
                    else fn(scores, path, axis=args.axis)  # type: ignore[call-arg]
                )
                log.info("wrote %s", written)
            except ValueError as exc:
                # A missing figure is worth a warning, not a failed run: the
                # tables are the deliverable and a partial sweep legitimately
                # lacks the fields a plot needs.
                log.warning("skipped %s: %s", label, exc)

    print(report)
    log.info("wrote %s", args.out / "results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
