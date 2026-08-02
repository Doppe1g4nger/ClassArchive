"""Run the equal-budget hyperparameter sweep for one method.

    iqssl-sweep --method simclr --data data/easy
    iqssl-sweep --method mae --data data/easy --dry-run   # print the command, run nothing

A thin driver rather than a second experiment framework. Hydra's Optuna sweeper
does the search; this assembles the invocation from the method's own ``search:``
block, and — the reason it exists at all — puts the fairness checks somewhere
they run *before* nine trials of compute rather than after:

* :func:`~iqssl.cli.pretrain.check_sweep_budget` refuses an unequal trial count.
* :func:`~iqssl.cli.pretrain.check_search_space` refuses a space that reaches
  for a knob the contract holds constant.
* ``experiment=hpo`` sets ``val_probe: true``, so what the sweeper maximizes is
  a validation probe score. Sweeping without it would optimize the pretraining
  loss, and BYOL and SimSiam reach near-zero loss exactly when they collapse.
* ``--data`` is required, because ``hpo`` pins no dataset. An objective is only
  a signal where the task is learnable; on a dataset where every method probes
  at chance, nine trials rank noise and still report a winner.

Doing this in argparse, outside Hydra, is deliberate: a driver that composed
itself from the same config tree it validates could be overridden into skipping
its own checks.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from iqssl.cli.pretrain import CONFIG_DIR, N_TRIALS, check_search_space, check_sweep_budget
from iqssl.registry import autodiscover
from iqssl.utils.logging_ import get_logger, setup_console_logging

log = get_logger(__name__)


def load_cfg(method: str, experiment: str, data_root: str | None = None) -> DictConfig:
    autodiscover()
    overrides = [f"method={method}", f"experiment={experiment}"]
    if data_root is not None:
        overrides.append(f"data.root={data_root}")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="pretrain", overrides=overrides)


def build_command(
    method: str,
    experiment: str,
    n_trials: int,
    extra: list[str],
    data_root: str | None = None,
) -> list[str]:
    """Assemble (and validate) the sweep invocation."""
    cfg = load_cfg(method, experiment, data_root)

    check_sweep_budget(n_trials)
    space = check_search_space(cfg)
    if not space:
        raise ValueError(
            f"method {method!r} declares no search: block, so there is nothing to "
            "sweep. `random` is the deliberate case -- an untrained encoder has no "
            "hyperparameters, and nine trials would measure nothing nine times."
        )
    if not cfg.get("val_probe", False):
        raise ValueError(
            f"experiment {experiment!r} does not set val_probe: true, so main() would "
            "return the pretraining loss and the sweeper would optimize that. For the "
            "negative-free methods that selects collapse. Use experiment=hpo."
        )
    if OmegaConf.is_missing(cfg.data, "root") or cfg.data.root is None:
        raise ValueError(
            f"experiment {experiment!r} names no dataset, so pass --data. Tuning has to "
            "run on the data the comparison reports: the val-probe objective is only a "
            "signal where the task is learnable, and on a dataset where every method "
            "probes at chance the sweep ranks noise and still names a winner."
        )

    cmd = [
        sys.executable,
        "-m",
        "iqssl.cli.pretrain",
        "--multirun",
        "hydra/sweeper=equal_budget",
        f"hydra.sweeper.n_trials={n_trials}",
        f"method={method}",
        f"experiment={experiment}",
    ]
    if data_root is not None:
        cmd.append(f"data.root={data_root}")
    cmd.append(sweeper_params_override(space))
    return cmd + extra


def sweeper_params_override(space: dict[str, str]) -> str:
    """The search space as one Hydra override, and every character matters.

    The Optuna sweeper wants ``params`` as a *flat* mapping from override string
    to space expression -- ``{"train.base_lr": "tag(log, interval(...))"}``. Four
    ways of saying that fail, each differently, and all four were found by
    running it rather than by reading it:

    * ``hydra.sweeper.params.train.base_lr=tag(log, interval(1e-4, 1e-2))``
      parses the value as a *sweep expression*, and sweep expressions are
      refused on ``hydra.*`` keys: "Sweeping over Hydra's configuration is not
      supported". Quoting the value fixes that.
    * ``hydra.sweeper.params.train.base_lr="..."`` then fails as an override of
      a key that does not exist, because equal_budget.yaml ships ``params: {}``.
    * ``+hydra.sweeper.params.train.base_lr="..."`` appends, but the dotted path
      builds a *nested* dict, ``{train: {base_lr: ...}}``. The sweeper then tries
      to parse ``train={'base_lr': ...}`` as an override and dies with "no viable
      alternative at input '{'base_lr''".
    * Quoting the key to keep it flat is not available: Hydra's override grammar
      has no production for a quoted key.

    What works is passing the whole mapping as one dict literal, whose keys may
    contain dots precisely because they are not a path. Values stay quoted so
    they arrive as strings; keys must stay *unquoted* or the grammar rejects
    them.
    """
    body = ", ".join(f'{k}: "{v}"' for k, v in space.items())
    return f"hydra.sweeper.params={{{body}}}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--method", required=True)
    p.add_argument("--experiment", default="hpo")
    p.add_argument(
        "--data",
        default=None,
        help="dataset root to tune on, e.g. data/easy. Required by experiment=hpo, "
        "which pins no dataset of its own so that inheriting the smoke default "
        "cannot happen silently",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"must equal the contracted budget ({N_TRIALS}); the flag exists so an "
        "attempt to change it fails loudly rather than silently",
    )
    p.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    p.add_argument("overrides", nargs="*", help="extra Hydra overrides")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging(logging.INFO)

    cmd = build_command(
        args.method, args.experiment, args.n_trials, args.overrides, data_root=args.data
    )
    printable = " ".join(cmd)
    if args.dry_run:
        print(printable)
        return 0

    log.info("sweeping %s: %d trials", args.method, args.n_trials)
    log.info("%s", printable)
    return subprocess.call(cmd, cwd=Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
