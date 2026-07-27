"""Content hashing for datasets and configs.

The dataset hash is what makes a comparison auditable six months later: every
run records it, and the aggregator refuses to pool runs whose hashes disagree.
Without this, a regenerated dataset silently invalidates a whole results table
and nothing in the pipeline notices.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Floats are formatted via ``repr`` by ``json``, which round-trips exactly for
    IEEE doubles, so a config hash is stable across processes and platforms.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_config(cfg: Any) -> str:
    """Hash a config mapping (DictConfig, dict, or dataclass-derived dict)."""
    if hasattr(cfg, "_content") or type(cfg).__name__ == "DictConfig":
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(cfg, resolve=True)
    return sha256_bytes(canonical_json(cfg).encode())


def dataset_hash(shard_hashes: dict[str, str], generator_cfg: Any, version: str) -> str:
    """Fold per-shard content hashes plus the generator config into one id.

    Includes the generator *version* so a change in generation semantics
    invalidates the hash even if by coincidence the bytes would match.
    """
    payload = {
        "version": version,
        "generator": (
            generator_cfg
            if isinstance(generator_cfg, dict)
            else json.loads(canonical_json(generator_cfg))
        ),
        "shards": dict(sorted(shard_hashes.items())),
    }
    return "sha256:" + sha256_bytes(canonical_json(payload).encode())


def short(h: str, n: int = 12) -> str:
    """Human-readable prefix for log lines and directory names."""
    return h.split(":", 1)[-1][:n]
