"""Rebuild a pretrained encoder from a run directory, and refuse the wrong data.

A run directory is self-describing: ``config_full.yaml`` holds the resolved
Hydra config the run was launched with, ``checkpoint.pt`` the method weights,
``config.json`` the dataset hash. Evaluation reconstructs the exact encoder and
method from those three files and nothing else, so a result can be reproduced
from the artifact without the shell history that produced it.
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from iqssl.data.build import load_manifest
from iqssl.utils.logging_ import get_logger, read_json

log = get_logger(__name__)


class DatasetMismatch(RuntimeError):
    """Raised when a run is evaluated against data it was not trained on."""


def load_run_config(run_dir: str | Path) -> DictConfig:
    path = Path(run_dir) / "config_full.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Runs older than the full-config snapshot cannot be "
            "evaluated: config.json records the method's name but not the encoder's "
            "arguments, and guessing the encoder would silently evaluate a "
            "different model."
        )
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return cfg


def check_dataset_hash(
    run_dir: str | Path, data_root: str | Path, *, allow_mismatch: bool = False
) -> str:
    """Compare the run's recorded dataset hash with the target dataset's.

    This is the pooling guarantee applied early: a probe score against different
    data is not a worse number, it is a number about something else, and by the
    time it sits in a results table nothing downstream can tell. Mismatches
    refuse loudly; ``allow_mismatch`` exists for debugging and stamps the log so
    the override cannot pass unnoticed.
    """
    recorded = read_json(Path(run_dir) / "config.json")["dataset_hash"]
    actual = load_manifest(data_root)["dataset_hash"]
    if recorded != actual:
        msg = (
            f"run {run_dir} was pretrained on {recorded} but {data_root} is {actual}. "
            "Evaluating across datasets produces a number about the wrong thing."
        )
        if not allow_mismatch:
            raise DatasetMismatch(msg)
        log.warning("OVERRIDDEN dataset-hash mismatch: %s", msg)
    return actual


def load_encoder(run_dir: str | Path, crop_len: int, n_classes: int) -> nn.Module:
    """Rebuild the method, load its weights, return the frozen eval encoder.

    Goes through ``method.encoder_for_eval()`` rather than ``.encoder``: BYOL
    documents a real choice there (student vs EMA teacher), and reaching around
    it would quietly re-open a degree of freedom the method closed.
    """
    from iqssl.cli.pretrain import build_method

    cfg = load_run_config(run_dir)
    method = build_method(cfg, crop_len, seed=int(cfg.train.seed), n_classes=n_classes)

    state = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu", weights_only=True)
    method.load_state_dict(state["method"])

    encoder = method.encoder_for_eval()
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder
