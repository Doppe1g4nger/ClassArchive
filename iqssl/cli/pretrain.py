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

import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from iqssl.data.dataset import IQDataset
from iqssl.eval.quick import format_scores, is_collapsed, objective_from, val_probe_score
from iqssl.registry import ENCODERS, METHODS, autodiscover
from iqssl.train.loop import TrainConfig, train
from iqssl.utils.logging_ import get_logger, setup_console_logging, write_json
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
    """OmegaConf container or plain mapping -> plain dict.

    Accepts plain dicts because `.get("train", {})` on a DictConfig returns the
    Python default, not a config node — so a method config *without* a train
    block handed OmegaConf.to_container a bare dict and crashed. Every method
    config shipped with one until `supervised`, which is how it went unnoticed.
    """
    if isinstance(node, dict):
        return dict(node)
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True) or {})


def cli_overridden_train_keys() -> set[str]:
    """Which ``train.*`` keys the caller set on the command line.

    Hydra records its task overrides, and the sweeper supplies its sampled
    hyperparameters the same way, so this covers both a human typing
    ``train.base_lr=2e-4`` and Optuna proposing one.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        task = HydraConfig.get().overrides.task
    except Exception:
        # Called outside a Hydra run (tests, notebooks): nothing was overridden.
        return set()

    keys = set()
    for item in task:
        stripped = str(item).lstrip("+~")
        if "=" in stripped:
            key = stripped.split("=", 1)[0]
            if key.startswith("train."):
                keys.add(key.split(".", 1)[1])
    return keys


def merge_train_config(cfg: DictConfig, cli_overrides: set[str] | None = None) -> TrainConfig:
    """Fold a method's permitted defaults into the experiment's training config.

    Precedence is root < method < experiment < command line, and the last term
    is the one this function exists to get right. A method's ``train:`` block is
    a *default* -- the published optimizer and learning rate it was designed
    with -- not a final say.

    Applying the method block unconditionally on top, as this did originally,
    silently discarded command-line overrides. That was not merely inconvenient:
    the HPO search spaces tune ``train.base_lr``, and the sweeper delivers its
    proposals as command-line overrides, so every one of the nine trials ran the
    method's static learning rate. The sweep would have reported a winner that
    differed from the others only by seed noise, while the budget check, the
    search-space validation and the val-probe objective all passed.
    """
    base = _as_dict(cfg.train)
    method_defaults = _as_dict(cfg.method.get("train", {}))

    illegal = set(method_defaults) - METHOD_TUNABLE
    if illegal:
        raise ValueError(
            f"method config {cfg.method.name!r} tries to set {sorted(illegal)}, which "
            f"the fairness contract holds constant across methods. A method may only "
            f"set {sorted(METHOD_TUNABLE)}; everything else belongs to the experiment."
        )

    explicit = cli_overridden_train_keys() if cli_overrides is None else cli_overrides
    applied = {k: v for k, v in method_defaults.items() if k not in explicit}
    for k in sorted(set(method_defaults) & explicit):
        log.info(
            "train.%s=%r from the command line overrides the %s default (%r)",
            k,
            base.get(k),
            cfg.method.name,
            method_defaults[k],
        )
    return TrainConfig(**{**base, **applied})


N_TRIALS = 9
"""The equal tuning budget, per the fairness contract.

Nine trials for every method, then three seeds at the winner. Enforced rather
than documented for the same reason METHOD_TUNABLE is: a method quietly given
thirty trials beats one given nine for reasons that have nothing to do with its
objective, and the resulting table looks entirely normal.
"""


def check_sweep_budget(n_trials: int) -> None:
    """Refuse a sweep whose trial count differs from the contracted budget."""
    if n_trials != N_TRIALS:
        raise ValueError(
            f"sweep requests {n_trials} trials but the fairness contract fixes the "
            f"budget at {N_TRIALS} per method. Equal budgets are what make 'we tuned "
            f"them fairly' a fact rather than a claim; change N_TRIALS if the budget "
            f"itself should change, which changes it for every method at once."
        )


def check_search_space(cfg: DictConfig) -> dict[str, str]:
    """Validate a method's ``search:`` block and return it.

    Every key must be a permitted training knob or one of *this* method's own
    constructor arguments. Without the check, a search space could reach for
    ``train.epochs`` and buy one method a longer schedule under the guise of
    tuning -- the same hole ``merge_train_config`` closes for static config.
    """
    space = _as_dict(cfg.method.get("search", {}))
    method_cls = METHODS.get(cfg.method.name)
    own_args = set(inspect.signature(method_cls.__init__).parameters)

    for key in space:
        if key.startswith("train."):
            knob = key.split(".", 1)[1]
            if knob not in METHOD_TUNABLE:
                raise ValueError(
                    f"search space for {cfg.method.name!r} tunes {key!r}, which the "
                    f"fairness contract holds constant. Tunable: "
                    f"{sorted('train.' + k for k in METHOD_TUNABLE)}."
                )
        elif key.startswith("method.args."):
            arg = key.split(".", 2)[2]
            if arg not in own_args:
                raise ValueError(
                    f"search space for {cfg.method.name!r} tunes {key!r}, which is not "
                    f"an argument of {method_cls.__name__}.__init__."
                )
        else:
            raise ValueError(
                f"search key {key!r} must start with 'train.' or 'method.args.'; "
                "anything else is outside what a method is allowed to vary."
            )
    return space


def build_method(cfg: DictConfig, seq_len: int, seed: int = 0, n_classes: int | None = None) -> Any:
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
    kwargs = _as_dict(cfg.method.args)

    # A method whose signature asks for n_classes gets the dataset's true count
    # unless the config pinned one explicitly. The class count is a dataset
    # property, not a method hyperparameter, and a config default that happened
    # to be smaller than the label range would fail only at the first unlucky
    # batch — deep inside cross_entropy, long after construction.
    if (
        n_classes is not None
        and "n_classes" not in kwargs
        and "n_classes" in inspect.signature(method_cls.__init__).parameters
    ):
        kwargs["n_classes"] = n_classes

    return method_cls(encoder, cfg, **kwargs)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="pretrain")
def main(cfg: DictConfig) -> float:
    setup_console_logging(logging.INFO)

    # `hpo` blanks data.root so that tuning cannot silently inherit the smoke
    # default; reached directly rather than through iqssl-sweep, say so here
    # instead of failing somewhere inside Path(None).
    if cfg.data.root is None:
        raise ValueError(
            f"experiment {cfg.experiment!r} names no dataset -- pass data.root=data/easy. "
            "It is left unset on purpose, so that a tuning run cannot quietly compose "
            "onto the root default and optimize a probe score that is chance."
        )

    dataset = IQDataset(
        cfg.data.root,
        "train",
        split_variant=cfg.data.split_variant,
        primary_label=cfg.data.primary_label,
        crop_len=cfg.data.crop_len,
    )

    method = build_method(
        cfg, dataset.crop_len, seed=cfg.train.seed, n_classes=dataset.num_primary_classes
    )
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

    # What this function *returns* is what Hydra's Optuna sweeper optimizes, so
    # it must be a measure of the representation and not of the objective value.
    # Returning the pretraining loss would be worse than uninformative: BYOL and
    # SimSiam reach near-zero loss precisely when they collapse, so a sweep would
    # select the collapsed configuration every time. Off by default because a
    # single run does not need it and the probe is not free.
    if not cfg.get("val_probe", False):
        return state.final_loss

    scores = val_probe_score(
        method.encoder_for_eval(),
        cfg.data.root,
        split_variant=cfg.data.split_variant,
        primary_label=cfg.data.primary_label,
        device=train_cfg.device,
        crop_len=cfg.data.crop_len,
    )
    if is_collapsed(scores):
        log.warning("collapse suspected -- %s", format_scores(scores))
    log.info("val probe: %s", format_scores(scores))
    write_json(out / "val_probe.json", scores)
    return objective_from(scores)


if __name__ == "__main__":
    main()
