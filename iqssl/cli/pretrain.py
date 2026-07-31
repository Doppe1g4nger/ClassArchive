"""Pretrain one method.

    iqssl-pretrain experiment=smoke_cpu method=simclr
    iqssl-pretrain -m experiment=main_comparison method=simclr,byol seed=0,1,2

Hydra here, unlike the argparse used for dataset construction, because this is
the command that gets swept: ``-m`` over method x seed x policy is the entire
experimental design, and composing that from config groups is what Hydra is for.

The run directory is keyed by experiment / method / seed / timestamp so a sweep
never writes two runs to one place, and every run records its dataset hash --
the aggregator refuses to pool runs whose hashes disagree.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from iqssl.data.dataset import IQDataset
from iqssl.registry import ENCODERS, METHODS, autodiscover
from iqssl.train.loop import TrainConfig, train
from iqssl.utils.logging_ import get_logger, setup_console_logging
from iqssl.utils.seed import seed_everything

log = get_logger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

METHOD_TUNABLE = frozenset({"optimizer", "base_lr", "weight_decay"})
"""The only training settings a method config may set.

This is the fairness contract made executable. Optimizer family, learning rate
and weight decay are *allowed* to vary per method -- forcing AdamW on SimSiam or
SGD on MAE would handicap them against their own published behaviour, and the
comparison would measure optimizer tolerance rather than objective quality.
Everything else -- epochs, batch size, warmup, grad clip, seed, augmentation
policy -- is held constant by the experiment, and a method config that reaches
for one of them is rejected rather than silently honoured.

Without this check the contract lives only in the README, and the first method
config to quietly ask for 200 epochs would invalidate the whole table.
"""


def _as_dict(node: Any) -> dict[str, Any]:
    """OmegaConf container -> plain dict, narrowed for the type checker."""
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True) or {})


def merge_train_config(cfg: DictConfig) -> TrainConfig:
    """Fold a method's permitted overrides into the experiment's training config."""
    base = _as_dict(cfg.train)
    overrides = _as_dict(cfg.method.get("train", {}))

    illegal = set(overrides) - METHOD_TUNABLE
    if illegal:
        raise ValueError(
            f"method config {cfg.method.name!r} tries to set {sorted(illegal)}, which "
            f"the fairness contract holds constant across methods. A method may only "
            f"set {sorted(METHOD_TUNABLE)}; everything else belongs to the experiment."
        )
    return TrainConfig(**{**base, **overrides})


def build_method(cfg: DictConfig, seq_len: int, seed: int = 0) -> Any:
    """Construct encoder then method.

    The encoder is built here rather than inside the method because it is the
    benchmark's control variable: every method must receive an architecturally
    identical encoder, and the only way to guarantee that is to build it in one
    place from one config that methods cannot override.

    ``seed_everything`` is called *immediately before* the encoder is built, and
    that placement is the point. The fairness contract holds "encoder
    architecture and init seed" constant, which means every method at seed 0 must
    start from the same weights -- not merely from the same architecture. Seeding
    only inside the training loop leaves initialization to whatever global RNG
    state the process happened to be in, so two runs of the same method at the
    same seed diverge, and two different methods at the same seed start from
    different points. Both were true before this line existed;
    ``tests/test_train.py`` now pins each.

    The method's own heads are then initialized from the same stream, so they are
    reproducible too -- but they legitimately differ between methods, since heads
    are part of the objective rather than the control variable.
    """
    autodiscover()
    seed_everything(seed)

    encoder_factory = ENCODERS.get(cfg.encoder.name)
    encoder = encoder_factory(seq_len=seq_len, **_as_dict(cfg.encoder.args))

    method_cls = METHODS.get(cfg.method.name)
    return method_cls(encoder, cfg, **_as_dict(cfg.method.args))


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="pretrain")
def main(cfg: DictConfig) -> float:
    setup_console_logging(logging.INFO)

    dataset = IQDataset(
        cfg.data.root,
        "train",
        split_variant=cfg.data.split_variant,
        primary_label=cfg.data.primary_label,
        crop_len=cfg.data.crop_len,
    )

    method = build_method(cfg, dataset.crop_len, seed=cfg.train.seed)
    train_cfg = merge_train_config(cfg)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(cfg.output_root) / cfg.experiment / cfg.method.name / f"seed{cfg.train.seed}" / stamp

    # The full resolved config, not just the summary in config.json: evaluation
    # rebuilds the exact encoder and method from the run directory alone, and
    # the summary records the method's *name* but not the encoder's arguments.
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=out / "config_full.yaml")

    state = train(method, dataset, train_cfg, out_dir=out, method_cfg=cfg)
    log.info("done: final loss %.4f, wrote %s", state.final_loss, out)
    return state.final_loss


if __name__ == "__main__":
    main()
