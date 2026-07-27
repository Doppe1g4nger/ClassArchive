"""Materialize a dataset to disk, with a manifest that makes it auditable.

Layout::

    <root>/MANIFEST.json        config, geometry, hashes, nuisance statistics
    <root>/meta.parquet         one row per sample: labels, nuisances, splits
    <root>/shard_0000.npy       (shard_size, 2, store_len) float32
    <root>/label_subsets.json   the shared few-label index lists

The manifest's ``dataset_hash`` is what lets the aggregator refuse to pool runs
built on different data. It is computed over the sample *bytes in global index
order*, not over the shard files, because how many files the samples were written
across is a storage decision: an otherwise identical dataset must not look like a
different one because someone passed a different ``--shard-size``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from iqssl.data import splits as split_mod
from iqssl.data.generator import SyntheticIQGenerator
from iqssl.data.params import (
    GENERATOR_VERSION,
    NUISANCE_FIELDS,
    GeneratorConfig,
)
from iqssl.utils.hashing import dataset_hash, sha256_file
from iqssl.utils.logging_ import get_logger, write_json

log = get_logger(__name__)

MANIFEST_NAME = "MANIFEST.json"
META_NAME = "meta.parquet"
SUBSETS_NAME = "label_subsets.json"

LABEL_FRACTIONS = (0.01, 0.1)


def build_dataset(
    cfg: GeneratorConfig,
    root: str | Path,
    *,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Generate, shard, split and hash a dataset. Returns the manifest."""
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass overwrite=True (or --overwrite) "
            "to replace it. Rebuilding in place would leave shards from two "
            "different generators side by side."
        )
    if root.exists() and overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    gen = SyntheticIQGenerator(cfg)
    n, shard_size = cfg.n_samples, cfg.shard_size

    content = hashlib.sha256()
    shard_files: dict[str, str] = {}
    metas: list[pd.DataFrame] = []

    for shard_i, start in enumerate(range(0, n, shard_size)):
        count = min(shard_size, n - start)
        iq, meta = gen.generate(start, count)

        # Fold the bytes in global index order, before they are distributed
        # across files, so the identity is of the *samples* not of the layout.
        content.update(np.ascontiguousarray(iq).tobytes())

        path = root / f"shard_{shard_i:04d}.npy"
        np.save(path, iq)
        shard_files[path.name] = sha256_file(path)
        metas.append(meta)

        if progress:
            log.info("shard %d: samples %d-%d", shard_i, start, start + count - 1)

    full = pd.concat(metas, ignore_index=True)
    full["shard"] = full.index.to_numpy() // shard_size
    full["row_in_shard"] = full.index.to_numpy() % shard_size

    stored_variants = _attach_splits(full, cfg.seed)
    subsets = {
        v: split_mod.make_label_subsets(
            full, full[f"split_{v}"].to_numpy(), LABEL_FRACTIONS, seed=cfg.seed
        )
        for v in stored_variants
    }
    write_json(root / SUBSETS_NAME, subsets)

    stats = _nuisance_stats(full)
    full.to_parquet(root / META_NAME, index=False)

    manifest = {
        "version": GENERATOR_VERSION,
        "dataset_hash": dataset_hash(
            {"content": content.hexdigest()}, cfg.identity(), GENERATOR_VERSION
        ),
        "n_samples": int(n),
        "store_len": int(gen.store_len),
        "crop_len": int(cfg.preset.crop_len),
        "n_emitters": int(cfg.preset.n_emitters),
        "modulations": list(cfg.modulations),
        "shard_size": int(shard_size),
        "shards": shard_files,
        "split_variants": stored_variants,
        "label_fractions": list(LABEL_FRACTIONS),
        "nuisance_stats": stats,
        "emitters": [e.to_dict() for e in gen.emitters],
        "config": cfg.identity() | {"n_samples": int(n), "shard_size": int(shard_size)},
    }
    write_json(root / MANIFEST_NAME, manifest)
    log.info("wrote %d samples to %s (%s)", n, root, manifest["dataset_hash"])
    return manifest


def _attach_splits(meta: pd.DataFrame, seed: int) -> list[str]:
    """Add a ``split_<variant>`` column per applicable variant.

    A variant that cannot be formed from this preset is skipped with a warning
    rather than written as a degenerate column. ``holdout_channel`` on a
    single-tap preset is the real case: it would otherwise store a split whose
    training side is empty, and nothing downstream would notice until a training
    run quietly saw no data.
    """
    stored = []
    for variant in split_mod.SPLIT_VARIANTS:
        try:
            meta[f"split_{variant}"] = split_mod.make_splits(meta, variant, seed=seed)
        except ValueError as exc:
            log.warning("skipping split variant %r: %s", variant, exc)
            continue
        stored.append(variant)
    return stored


def _nuisance_stats(meta: pd.DataFrame) -> dict:
    """Mean and std of each nuisance field, over the **training** split only.

    Standardizing with statistics computed over the whole dataset would leak test
    -- mildly, but for free, and in a benchmark whose entire claim is a controlled
    comparison.
    """
    train = meta[meta["split_iid"] == "train"]
    mean = [float(train[f].mean()) for f in NUISANCE_FIELDS]
    std = [float(train[f].std()) for f in NUISANCE_FIELDS]

    for f, s in zip(NUISANCE_FIELDS, std, strict=True):
        if not s > 0:
            raise ValueError(
                f"nuisance field {f!r} is constant across the training split, so "
                "standardizing it would divide by zero. Every field in "
                "NUISANCE_FIELDS must be drawn for every sample under every preset."
            )
    return {"fields": list(NUISANCE_FIELDS), "mean": mean, "std": std}


def load_manifest(root: str | Path) -> dict:
    with open(Path(root) / MANIFEST_NAME) as fh:
        return json.load(fh)


def verify_dataset(root: str | Path, *, quiet: bool = False) -> bool:
    """Re-hash every shard against the manifest. Returns True when all match."""
    root = Path(root)
    manifest = load_manifest(root)
    ok = True
    for name, expected in manifest["shards"].items():
        path = root / name
        if not path.exists():
            ok = False
            if not quiet:
                log.error("missing shard %s", name)
            continue
        actual = sha256_file(path)
        if actual != expected:
            ok = False
            if not quiet:
                log.error("shard %s hash mismatch: expected %s, got %s", name, expected, actual)
    return ok
