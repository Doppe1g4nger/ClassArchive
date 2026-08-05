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
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from iqssl.cli.pretrain import CONFIG_DIR, N_TRIALS, check_search_space, check_sweep_budget
from iqssl.registry import autodiscover
from iqssl.utils.logging_ import get_logger, read_json, setup_console_logging, write_json

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

    # Validates the *contract* (nine per method); how many still need running
    # is computed below from what the persisted study already holds.
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
        f"hydra.sweeper.n_trials={trials_to_run(method, n_trials)}",
        f"method={method}",
        f"experiment={experiment}",
    ]
    if data_root is not None:
        cmd.append(f"data.root={data_root}")
    # Persist the study so a killed sweep resumes instead of restarting.
    # equal_budget.yaml ships `storage: null`, which keeps everything in memory
    # -- and since check_sweep_budget requires all nine trials, a partial study
    # cannot legitimately name a winner, so an interruption costs the whole
    # method. A container suspension cost `barlow`'s five completed trials once
    # already. Needs the sqlalchemy<2 pin in the `sweep` extra; see pyproject.
    #
    # One database, one study per method: a shared study_name would pool eleven
    # methods' trials into one search over incompatible spaces.
    cmd.append(f"hydra.sweeper.storage=sqlite:///{STUDY_DB}")
    cmd.append(f"hydra.sweeper.study_name={method}")
    cmd.append(sweeper_params_override(space))
    return cmd + extra


def completed_trials(method: str, db: Path | None = None) -> int:
    """How many trials this method's persisted study already finished."""
    path = db or STUDY_DB
    if not path.exists():
        return 0
    try:
        import optuna

        study = optuna.load_study(study_name=method, storage=f"sqlite:///{path}")
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    except Exception as exc:
        # An unreadable or study-less database means "nothing done yet". Running
        # the full budget is the safe direction: too few trials would break the
        # contract silently, an unnecessary re-run only costs time.
        log.warning("could not read study for %s (%s); assuming no completed trials", method, exc)
        return 0


def trials_to_run(method: str, budget: int = N_TRIALS, db: Path | None = None) -> int:
    """Trials still owed, so an interrupted sweep finishes its budget exactly.

    Persistence alone does *not* give resumption -- it gives accumulation.
    Hydra's Optuna sweeper reads ``n_trials`` as "run this many **new** trials",
    so re-invoking a nine-trial sweep on a study that already holds nine runs
    nine more. A method interrupted once would end up with 14 or 18 trials while
    every other method got 9, which is exactly the inequality
    ``check_sweep_budget`` exists to prevent -- reintroduced, and silently, by
    the very change meant to make interruptions harmless.

    The contract is "nine trials per method", not "nine per invocation". This
    computes the difference so the two agree however many times a sweep is
    restarted.
    """
    return max(0, budget - completed_trials(method, db))


STUDY_DB = Path("outputs_sweep/hpo.db")
"""Optuna's persisted studies. One file, one study per method.

This is for *resumption*. Recovering the winner for downstream use goes through
:func:`save_best_params`, which reads run artifacts instead -- deliberately, so
that a future dependency change cannot strand results the way this one nearly
did. The two paths answer different questions and neither replaces the other.
"""


TUNED_DIR = Path("results/tuned")
"""Where the winner of each sweep is written, for the comparison to consume."""

TUNABLE_KEYS = ("optimizer", "base_lr", "weight_decay")
"""Which ``train.*`` values are carried from a sweep into the comparison.

The same three ``METHOD_TUNABLE`` permits. Anything else a trial happened to
record is held constant by the fairness contract and must not travel.

Method-specific coefficients travel too, but separately -- see
:func:`_searched_method_args`. They are not in this tuple because they are not
``train.*`` keys and are not shared across methods.
"""


def _searched_method_args(full: DictConfig) -> dict[str, Any]:
    """The method arguments this trial's search space actually tuned.

    A sweep tunes more than the learning rate. SupCon searches ``temperature``,
    MAE and TS-JEPA search ``mask_ratio``, and the winning trial's value for
    those is part of what won. Recovering only ``train.*`` would hand the
    comparison the tuned learning rate beside the *published* coefficient --
    a configuration no trial ever ran, and therefore one with no evidence behind
    it at all. On `supcon` that is not hypothetical: the winner used
    ``temperature`` 0.056 against a published 0.1.

    Filtered by the declared search space rather than taken wholesale, so the
    file records what tuning chose and not the untuned defaults it sat beside.
    ``check_search_space`` has already refused any key that is not a real
    constructor argument of this method, so the space is a safe filter.
    """
    search = full.get("method", {}).get("search") or {}
    args = full.get("method", {}).get("args") or {}
    prefix = "method.args."
    keys = [k[len(prefix) :] for k in search if k.startswith(prefix)]
    return {k: OmegaConf.to_object(args)[k] for k in keys if k in args}  # type: ignore[index]


def save_best_params(
    method: str, sweep_root: Path = Path("outputs/hpo"), out_dir: Path = TUNED_DIR
) -> dict[str, Any] | None:
    """Recover the winning trial from the run directories it left behind.

    Optuna's own study would be the obvious source, and is not usable here:
    ``equal_budget.yaml`` ships ``storage: null`` so the study dies with the
    process, and switching it to SQLite *breaks the sweep outright* -- optuna
    2.10.1 predates SQLAlchemy 2.x and cannot stamp its schema-version table, so
    every trial aborts before it starts. Pinning either side risks the sweeper
    plugin or the rest of the stack.

    Reading the artifacts avoids the dependency question entirely. Every trial
    already writes ``val_probe.json`` (the objective the sweeper maximized) and
    ``config.json`` (what produced it), so the winner is recoverable from disk
    with no extra machinery -- and recoverable *after the fact*, including from
    sweeps that ran before this function existed.
    """
    trials: list[dict[str, Any]] = []
    for probe_path in sorted((sweep_root / method).rglob("val_probe.json")):
        run = probe_path.parent
        cfg_path = run / "config.json"
        if not cfg_path.exists():
            continue
        probe, cfg = read_json(probe_path), read_json(cfg_path)
        score = probe.get("val_probe_acc")
        if score is None:
            continue
        train = cfg.get("train", {})
        # config.json records `train` but not `method.args`, so the tuned
        # coefficients come from the full config snapshot beside it. Absent for
        # runs predating that snapshot; those still yield their train params.
        full_path = run / "config_full.yaml"
        method_args = (
            _searched_method_args(OmegaConf.load(full_path))  # type: ignore[arg-type]
            if full_path.exists()
            else {}
        )
        trials.append(
            {
                "val_probe_acc": float(score),
                "params": {k: train[k] for k in TUNABLE_KEYS if k in train},
                "method_args": method_args,
                "run": str(run),
            }
        )

    if not trials:
        log.warning(
            "no completed trials for %s under %s -- nothing to tune with", method, sweep_root
        )
        return None

    best = max(trials, key=lambda t: float(t["val_probe_acc"]))
    scores: list[float] = sorted((float(t["val_probe_acc"]) for t in trials), reverse=True)
    payload = {
        "method": method,
        "best_params": best["params"],
        "best_method_args": best["method_args"],
        "best_value": best["val_probe_acc"],
        "best_run": best["run"],
        "n_trials": len(trials),
        "objective": "val_probe_acc",
        # The runner-up, recorded because on `supervised` the seven healthy
        # trials spanned 0.009 total. A winner that close to second place is a
        # selection, not a measurement, and the gap is the reader's warning.
        "runner_up_value": scores[1] if len(scores) > 1 else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{method}.json", payload)
    log.info(
        "tuned %s: %s (val_probe %.4f, runner-up %s, %d trials)",
        method,
        best["params"],
        best["val_probe_acc"],
        f"{scores[1]:.4f}" if len(scores) > 1 else "n/a",
        len(trials),
    )
    return payload


def tuned_overrides(
    method: str, tuned_dir: Path = TUNED_DIR, *, required: bool = True
) -> list[str]:
    """The banked winner for ``method``, as Hydra command-line overrides.

    This is the half of tuning that turns it from a number in a file into an
    experiment. Without it `results/tuned/` is write-only: the comparison
    composes each method's *published* defaults from its yaml, every run looks
    healthy, every table renders, and the claim "tuned per method on an equal
    budget" is simply false. On `supcon` that would have meant base_lr 0.3
    instead of 0.686 and temperature 0.1 instead of 0.056 -- 2.3x off on one
    axis and 1.8x on the other, with nothing anywhere reporting a problem.

    Missing files raise rather than falling back, for the same reason. A silent
    fallback is indistinguishable from success in every artifact the run
    produces, and the fallback is precisely the untuned configuration the
    comparison exists to avoid. Pass ``required=False`` only where an untuned
    baseline is the intent -- `random` has no search space and never sweeps.

    Overrides, not a merged config, because the command line is the top of
    `merge_train_config`'s precedence chain. Anything below it can be quietly
    overwritten by a method's own ``train:`` block -- which is exactly the bug
    that silently ran all nine trials of every early sweep at the static
    published learning rate.
    """
    path = tuned_dir / f"{method}.json"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"no tuned parameters for {method!r} at {path}. Run "
                f"`iqssl-sweep --method {method}` first, or pass required=False to "
                f"run it at its published defaults -- but then it is untuned, and "
                f"no table may describe it otherwise."
            )
        log.warning("no tuned parameters for %s; using published defaults", method)
        return []

    payload = read_json(path)
    overrides = [f"train.{k}={_override_value(v)}" for k, v in payload["best_params"].items()]
    overrides += [
        f"method.args.{k}={_override_value(v)}"
        for k, v in (payload.get("best_method_args") or {}).items()
    ]
    return overrides


def _override_value(v: Any) -> str:
    """Render a value so Hydra's parser reads back exactly what was tuned.

    `repr` on a float, not `str` or a format spec: `1.94e-07` must survive with
    every digit, and a rounded learning rate is a different experiment from the
    one the sweep selected.
    """
    return repr(v) if isinstance(v, float) else str(v)


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

    * ``hydra.sweeper.params={train.base_lr: "..."}`` gets the shape right but
      the verb wrong: merging new keys into ``params: {}`` is an *addition*, and
      struct mode refuses additions phrased as overrides.

    What works is that dict literal with ``+``: one mapping, keys containing
    dots precisely because they are not a path, added rather than overridden.
    Values stay quoted so they arrive as strings; keys must stay *unquoted* or
    the grammar rejects them.
    """
    body = ", ".join(f'{k}: "{v}"' for k, v in space.items())
    return f"+hydra.sweeper.params={{{body}}}"


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
    code = subprocess.call(cmd, cwd=Path.cwd())

    # Bank the winner even on a nonzero exit: a sweep that completed eight of
    # nine trials still has a best trial worth keeping, and losing it would
    # mean re-running the whole hour.
    save_best_params(args.method)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
