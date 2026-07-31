"""Evaluate a pretrained run through the fixed protocol.

    iqssl-evaluate --run outputs/smoke_cpu/simclr/seed0/<timestamp>
    iqssl-evaluate --run <dir> --data data/easy --device cuda

argparse rather than Hydra, deliberately. The protocol is part of the fairness
contract: probe budgets, kNN parameters and the ridge penalty are module
constants in :mod:`iqssl.eval.probes`, applied identically to every method. A
config tree here would invite exactly the per-method drift the benchmark exists
to prevent, so the CLI exposes only *which* run, *which* dataset, *which*
device — never how hard to probe.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from iqssl.eval.protocol import evaluate_run
from iqssl.utils.logging_ import setup_console_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, type=Path, help="pretraining run directory")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="dataset root; defaults to the one recorded in the run's config",
    )
    p.add_argument("--split-variant", default="iid")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--skip-finetune",
        action="store_true",
        help="smoke iterations only -- a report without the finetune column is not the protocol",
    )
    p.add_argument(
        "--allow-hash-mismatch",
        action="store_true",
        help="DEBUG ONLY: evaluate against data the run was not pretrained on",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging(logging.INFO)

    report = evaluate_run(
        args.run,
        args.data,
        split_variant=args.split_variant,
        device=args.device,
        allow_hash_mismatch=args.allow_hash_mismatch,
        skip_finetune=args.skip_finetune,
    )

    print(f"\nEvaluation -- {report['method']} ({report['run']})")
    print(f"  dataset  {report['dataset_hash']}")
    for axis, res in report["axes"].items():
        print(f"\n  [{axis}]")
        print(f"  {'fraction':>10} {'linear':>8} {'knn':>8} {'finetune':>9}")
        for frac in report["label_fractions"]:
            key = f"{frac:g}"
            ft = res["finetune"].get(key)
            print(
                f"  {key:>10} {res['linear_probe'][key]:>8.3f} {res['knn'][key]:>8.3f} "
                f"{ft if ft is None else format(ft, '.3f'):>9}"
            )
    print("\n  nuisance R^2 (near 0 = discarded, near 1 = retained):")
    for field, r2 in report["nuisance_r2"].items():
        print(f"    {field:<24} {r2:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
