"""Walk a directory of runs into one tidy frame.

A run is a directory holding ``eval.json`` (what the representation scored),
``config.json`` (what was held constant) and ``summary.json`` (what it cost).
Runs without ``eval.json`` are skipped with a warning rather than silently
dropped: an unevaluated run in a sweep directory usually means a crashed
evaluation, and quietly omitting it would shrink a seed group from three to two
without anything saying so.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from iqssl.utils.logging_ import get_logger, read_json

log = get_logger(__name__)


def run_metadata(run_dir: Path) -> dict:
    """The fields the fairness invariants are checked against."""
    cfg = read_json(run_dir / "config.json")
    train = cfg.get("train", {})
    meta = {
        "run": str(run_dir),
        "method": cfg.get("method"),
        "dataset_hash": cfg.get("dataset_hash"),
        "seed": train.get("seed"),
        "epochs": train.get("epochs"),
        "batch_size": train.get("batch_size"),
        "warmup_frac": train.get("warmup_frac"),
        "grad_clip": train.get("grad_clip"),
        "augment": train.get("augment"),
        "optimizer": train.get("optimizer"),
        "base_lr": train.get("base_lr"),
        "weight_decay": train.get("weight_decay"),
    }

    full = run_dir / "config_full.yaml"
    if full.exists():
        from omegaconf import OmegaConf

        resolved = OmegaConf.load(full)
        meta["encoder"] = OmegaConf.to_container(resolved.encoder, resolve=True)
        meta["crop_len"] = resolved.data.get("crop_len")

    summary = run_dir / "summary.json"
    if summary.exists():
        meta.update({k: v for k, v in read_json(summary).items() if k.startswith("compute/")})
    return meta


def collect_runs(root: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """Return ``(tidy_scores, run_metadata)`` for every evaluated run under ``root``.

    The tidy frame carries one row per (run, method, seed, axis, probe,
    fraction, score) so every table downstream is a groupby rather than a
    bespoke traversal of nested JSON.
    """
    root = Path(root)
    rows: list[dict] = []
    metas: list[dict] = []

    for eval_path in sorted(root.rglob("eval.json")):
        run_dir = eval_path.parent
        report = read_json(eval_path)
        meta = run_metadata(run_dir)
        # eval.json always carries the registry key; config.json did not until
        # recently, so prefer the report and let older runs still aggregate.
        meta["method"] = report.get("method") or meta["method"]
        meta["split_variant"] = report.get("split_variant")
        meta["finetune_included"] = report.get("finetune_included", True)
        metas.append(meta)

        for axis, res in report["axes"].items():
            for probe in ("linear_probe", "knn", "finetune"):
                for fraction, score in res.get(probe, {}).items():
                    rows.append(
                        {
                            "run": str(run_dir),
                            "method": meta["method"],
                            "seed": meta["seed"],
                            "axis": axis,
                            "probe": probe,
                            "fraction": float(fraction),
                            "score": float(score),
                        }
                    )
            for band in res.get("snr_quartiles", []):
                rows.append(
                    {
                        "run": str(run_dir),
                        "method": meta["method"],
                        "seed": meta["seed"],
                        "axis": axis,
                        "probe": "snr_band",
                        "fraction": float(band["snr_lo"]),
                        "score": float(band["accuracy"]),
                    }
                )

        for field, r2 in report.get("nuisance_r2", {}).items():
            rows.append(
                {
                    "run": str(run_dir),
                    "method": meta["method"],
                    "seed": meta["seed"],
                    "axis": "nuisance",
                    "probe": field,
                    "fraction": float("nan"),
                    "score": float(r2),
                }
            )

    unevaluated = [
        d for d in sorted(root.rglob("checkpoint.pt")) if not (d.parent / "eval.json").exists()
    ]
    for d in unevaluated:
        log.warning(
            "run %s has a checkpoint but no eval.json -- skipped. An unevaluated run "
            "in a sweep usually means evaluation crashed, and dropping it silently "
            "would shrink a seed group without saying so.",
            d.parent,
        )

    return pd.DataFrame(rows), metas
