"""Build a synthetic dataset to disk.

    iqssl-build-dataset --difficulty medium --n-samples 200000 --out data/synth_v1
    iqssl-build-dataset --difficulty smoke --n-samples 2048 --out data/smoke

Plain argparse rather than Hydra: dataset construction is a one-off that has to
work before any training config exists, and its parameters are the dataset's
identity — they belong in the stored manifest, not in a composed config tree.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from iqssl.data.build import build_dataset, verify_dataset
from iqssl.data.params import PRESETS, GeneratorConfig
from iqssl.utils.logging_ import setup_console_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--difficulty", default="medium", choices=sorted(PRESETS))
    p.add_argument("--n-samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=20_000)
    p.add_argument("--span", type=int, default=11, help="RRC span in symbols (odd)")
    p.add_argument("--normalize", default="rms", choices=["rms", "peak", "none"])
    p.add_argument("--modulations", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true", help="re-hash shards after building")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging(logging.WARNING if args.quiet else logging.INFO)

    cfg = GeneratorConfig(
        n_samples=args.n_samples,
        difficulty=args.difficulty,
        seed=args.seed,
        shard_size=args.shard_size,
        span=args.span,
        normalize=args.normalize,
        modulations=(tuple(args.modulations) if args.modulations else GeneratorConfig.modulations),
    )
    manifest = build_dataset(cfg, args.out, overwrite=args.overwrite, progress=not args.quiet)

    if args.verify and not verify_dataset(args.out):
        print("dataset verification FAILED", file=sys.stderr)
        return 1

    print(f"\ndataset_hash: {manifest['dataset_hash']}")
    print(f"wrote {manifest['n_samples']} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
