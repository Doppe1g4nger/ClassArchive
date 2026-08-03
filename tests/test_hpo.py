"""Stage-8 tests: the tuning objective and the equal budget.

Both properties here are the kind that never fail loudly. A sweep that selects
on the test split still produces a plausible table; so does one that gives a
method thirty trials. Neither is detectable after the fact from the numbers, so
they are asserted before the compute is spent.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from omegaconf import OmegaConf

from iqssl.cli.pretrain import (
    METHOD_TUNABLE,
    N_TRIALS,
    check_search_space,
    check_sweep_budget,
)
from iqssl.registry import METHODS, autodiscover

autodiscover()


class TestBudget:
    def test_contracted_budget_is_accepted(self):
        check_sweep_budget(N_TRIALS)

    @pytest.mark.parametrize("n", [1, 8, 10, 30])
    def test_any_other_budget_is_refused(self, n):
        with pytest.raises(ValueError, match="fairness contract fixes the budget"):
            check_sweep_budget(n)


class TestSearchSpaces:
    def _cfg(self, method: str, space: dict):
        return OmegaConf.create({"method": {"name": method, "search": space}})

    def test_tunable_training_knobs_are_allowed(self):
        space = {f"train.{k}": "interval(0, 1)" for k in METHOD_TUNABLE if k != "optimizer"}
        assert check_search_space(self._cfg("simclr", space)) == space

    def test_held_constant_knobs_are_refused(self):
        """The hole merge_train_config closes for static config, closed again
        for the dynamic one: a search space could otherwise buy a method a
        longer schedule under the guise of tuning."""
        for knob in ("epochs", "batch_size", "seed", "augment", "grad_clip"):
            with pytest.raises(ValueError, match="holds constant"):
                check_search_space(self._cfg("simclr", {f"train.{knob}": "choice(1, 2)"}))

    def test_method_own_arguments_are_allowed(self):
        space = {"method.args.temperature": "interval(0.05, 0.5)"}
        assert check_search_space(self._cfg("simclr", space)) == space

    def test_arguments_of_a_different_method_are_refused(self):
        # `lambd` is Barlow's, not SimCLR's; without the check it would be
        # accepted, passed to a constructor that has no such parameter, and
        # crash nine trials deep.
        with pytest.raises(ValueError, match="not an argument"):
            check_search_space(self._cfg("simclr", {"method.args.lambd": "interval(0, 1)"}))

    def test_unprefixed_keys_are_refused(self):
        with pytest.raises(ValueError, match="must start with"):
            check_search_space(self._cfg("simclr", {"lr": "interval(0, 1)"}))

    @pytest.mark.parametrize("name", sorted(METHODS.keys()))
    def test_every_shipped_search_space_is_legal(self, name):
        """Validates the configs as shipped, so an illegal space cannot reach a
        sweep. `random` legitimately declares none."""
        from hydra import compose, initialize_config_dir

        from iqssl.cli.pretrain import CONFIG_DIR

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="pretrain", overrides=[f"method={name}"])
        space = check_search_space(cfg)
        assert space or name == "random"


class TestValProbeObjective:
    def test_the_probe_never_opens_the_test_split(self, monkeypatch, tmp_path):
        """The leak that would invalidate every headline number, silently.

        Nine trials across twelve methods selecting on test would tune the whole
        benchmark against the split it reports. Rather than trusting the source
        to keep saying "val", this records which splits get opened.
        """
        from iqssl.data.build import build_dataset
        from iqssl.data.params import GeneratorConfig

        build_dataset(
            GeneratorConfig(n_samples=512, shard_size=512, difficulty="smoke", seed=0),
            tmp_path,
            overwrite=True,
            progress=False,
        )

        import iqssl.eval.quick as quick
        from iqssl.data.dataset import IQDataset

        opened: list[str] = []
        real = IQDataset.__init__

        def spy(self, root, split="train", **kw):
            opened.append(split)
            real(self, root, split, **kw)

        monkeypatch.setattr(IQDataset, "__init__", spy)

        from iqssl.models.vit1d import vit1d

        scores = quick.val_probe_score(vit1d(size="tiny", seq_len=256, patch_size=16), tmp_path)

        assert "test" not in opened, f"tuning opened the test split: {opened}"
        assert set(opened) == {"train", "val"}
        assert 0.0 <= scores["val_probe_acc"] <= 1.0
        assert scores["chance"] == pytest.approx(1 / 8)

    def test_objective_is_the_probe_accuracy(self):
        from iqssl.eval.quick import objective_from

        scores = {"val_probe_acc": 0.42, "val_knn_acc": 0.1, "val_rankme": 9.0}
        assert objective_from(scores) == 0.42

    def test_collapse_is_flagged_not_raised(self):
        """A collapsed trial should lose on its score, not abort the sweep: low
        effective rank is legitimate early in a short tuning schedule."""
        from iqssl.eval.quick import is_collapsed

        assert is_collapsed({"val_rankme": 1.0})
        assert is_collapsed({"val_rankme": float("nan")})
        assert not is_collapsed({"val_rankme": 40.0})


class TestSweepDriver:
    def test_command_includes_budget_and_space(self):
        from iqssl.cli.sweep import build_command

        cmd = " ".join(build_command("simclr", "hpo", N_TRIALS, [], data_root="data/easy"))
        assert f"hydra.sweeper.n_trials={N_TRIALS}" in cmd
        assert "hydra/sweeper=equal_budget" in cmd
        assert "method.args.temperature" in cmd
        assert "data.root=data/easy" in cmd

    def test_search_space_values_survive_hydras_override_parser(self):
        """The bug that made `iqssl-sweep` fail on its first real invocation.

        Hydra parses `tag(log, interval(1e-4, 1e-2))` as a sweep expression, and
        refuses sweep expressions on `hydra.*` keys -- so the driver aborted
        before trial one with "Sweeping over Hydra's configuration is not
        supported". Quoting keeps it a string for the Optuna sweeper to parse.
        Dry-running the command could never have caught this: the text printed
        was correct, and only Hydra's own parser rejects it.
        """
        from hydra.core.override_parser.overrides_parser import OverridesParser

        from iqssl.cli.sweep import build_command

        cmd = build_command("simclr", "hpo", N_TRIALS, [], data_root="data/easy")
        params = [c for c in cmd if c.startswith("+hydra.sweeper.params=")]
        assert len(params) == 1, f"expected one appended params override, got {params}"

        override = OverridesParser.create().parse_overrides(params)[0]
        assert not override.is_sweep_override(), (
            "parses as a sweep over hydra config, which Hydra refuses outright"
        )

        value = override.value()
        assert isinstance(value, dict)
        # Flat, dotted keys -- not {train: {base_lr: ...}}. The sweeper reads
        # each key as an override string, so a nested dict makes it try to parse
        # `train={'base_lr': ...}` and fail in its own grammar.
        assert "train.base_lr" in value, f"search space is not flat: {dict(value)}"
        assert all(isinstance(v, str) for v in value.values()), (
            "space expressions must arrive as strings for the sweeper to parse"
        )

    def test_refuses_to_tune_without_a_named_dataset(self):
        """`hpo` pins no `data:` group, and an experiment that names none composes
        onto the root default -- `smoke`, where the difficulty gate showed every
        method at chance. Nine trials would maximize noise and report a winner."""
        from iqssl.cli.sweep import build_command

        with pytest.raises(ValueError, match="names no dataset"):
            build_command("simclr", "hpo", N_TRIALS, [])

    def test_refuses_an_experiment_without_the_val_probe(self):
        """Sweeping smoke_cpu would optimize the pretraining loss, which for the
        negative-free methods selects collapse."""
        from iqssl.cli.sweep import build_command

        with pytest.raises(ValueError, match="val_probe"):
            build_command("simclr", "smoke_cpu", N_TRIALS, [])

    def test_refuses_a_method_with_nothing_to_tune(self):
        from iqssl.cli.sweep import build_command

        with pytest.raises(ValueError, match="nothing to sweep"):
            build_command("random", "hpo", N_TRIALS, [])


class TestOverridePrecedence:
    """A method's `train:` block is a default, not a final say.

    This is the bug that would have made the entire HPO stage a no-op. The
    search spaces tune `train.base_lr`; the sweeper delivers proposals as
    command-line overrides; the method block was applied unconditionally on top.
    All nine trials would have run the method's static learning rate and the
    sweep would have named a winner that differed only by seed noise -- with the
    budget check, the search-space validation and the val-probe objective all
    reporting success. Nothing downstream could have detected it.
    """

    def _cfg(self, cli_base_lr: float):
        return OmegaConf.create(
            {
                "train": {"epochs": 10, "optimizer": "adamw", "base_lr": cli_base_lr},
                "method": {"name": "simclr", "train": {"optimizer": "lars", "base_lr": 0.3}},
            }
        )

    def test_command_line_beats_the_method_default(self):
        from iqssl.cli.pretrain import merge_train_config

        merged = merge_train_config(self._cfg(2e-4), cli_overrides={"base_lr"})
        assert merged.base_lr == 2e-4, "the sweeper's proposal must reach the loop"
        # Untouched keys still take the method's published default.
        assert merged.optimizer == "lars"

    def test_method_default_applies_when_nothing_was_overridden(self):
        from iqssl.cli.pretrain import merge_train_config

        merged = merge_train_config(self._cfg(1e-3), cli_overrides=set())
        assert merged.base_lr == 0.3
        assert merged.optimizer == "lars"

    def test_experiment_settings_are_untouched_either_way(self):
        from iqssl.cli.pretrain import merge_train_config

        for overrides in (set(), {"base_lr"}):
            assert merge_train_config(self._cfg(2e-4), cli_overrides=overrides).epochs == 10

    def test_override_keys_are_parsed_from_hydra_task_overrides(self, monkeypatch):
        """Covers the real path: what the sweeper actually hands Hydra."""
        import iqssl.cli.pretrain as mod

        class _Overrides:
            task: ClassVar[list[str]] = [
                "method=simclr",
                "train.base_lr=0.05",
                "+train.weight_decay=1e-6",
                "seed=1",
            ]

        class _Cfg:
            overrides = _Overrides()

        class _HydraConfig:
            @staticmethod
            def get():
                return _Cfg()

        monkeypatch.setitem(
            __import__("sys").modules,
            "hydra.core.hydra_config",
            type("m", (), {"HydraConfig": _HydraConfig}),
        )
        assert mod.cli_overridden_train_keys() == {"base_lr", "weight_decay"}

    def test_no_hydra_context_means_nothing_was_overridden(self):
        from iqssl.cli.pretrain import cli_overridden_train_keys

        assert cli_overridden_train_keys() == set()


class TestTunedParamsSurviveTheSweep:
    """Stage 8 has to be able to hand its answer to Stage 9.

    It could not: `equal_budget.yaml` ships `storage: null`, so Optuna's study
    lives in memory and the winner reaches only stdout. Switching to SQLite makes
    it worse -- optuna 2.10.1 predates SQLAlchemy 2.x, cannot stamp its
    schema-version table, and every trial aborts before starting. So the winner
    is recovered from the artifacts each trial already writes.
    """

    def _trial(self, root, method, stamp, probe, base_lr):
        import json

        d = root / method / "seed0" / stamp
        d.mkdir(parents=True)
        (d / "val_probe.json").write_text(json.dumps({"val_probe_acc": probe}))
        (d / "config.json").write_text(
            json.dumps(
                {
                    "train": {
                        "optimizer": "adamw",
                        "base_lr": base_lr,
                        "weight_decay": 1e-4,
                        "epochs": 3,
                        "batch_size": 128,
                    }
                }
            )
        )

    def test_the_highest_scoring_trial_wins(self, tmp_path):
        from iqssl.cli.sweep import save_best_params

        for stamp, probe, lr in (("a", 0.11, 1e-4), ("b", 0.19, 5e-4), ("c", 0.15, 2e-3)):
            self._trial(tmp_path, "simclr", stamp, probe, lr)

        got = save_best_params("simclr", sweep_root=tmp_path, out_dir=tmp_path / "tuned")
        assert got["best_value"] == pytest.approx(0.19)
        assert got["best_params"]["base_lr"] == pytest.approx(5e-4)
        assert got["n_trials"] == 3

    def test_only_contract_tunable_keys_travel(self, tmp_path):
        """A trial's config records the whole training block. Carrying `epochs`
        or `batch_size` into the comparison would buy one method a different
        schedule -- the exact hole METHOD_TUNABLE exists to close."""
        from iqssl.cli.sweep import save_best_params

        self._trial(tmp_path, "byol", "a", 0.2, 1e-3)
        got = save_best_params("byol", sweep_root=tmp_path, out_dir=tmp_path / "tuned")
        assert set(got["best_params"]) <= METHOD_TUNABLE
        assert "epochs" not in got["best_params"]
        assert "batch_size" not in got["best_params"]

    def test_the_runner_up_is_recorded(self, tmp_path):
        """On `supervised` the seven healthy trials spanned 0.009. A winner that
        close to second place is a selection, not a measurement."""
        from iqssl.cli.sweep import save_best_params

        self._trial(tmp_path, "mae", "a", 0.140, 1e-4)
        self._trial(tmp_path, "mae", "b", 0.141, 5e-4)
        got = save_best_params("mae", sweep_root=tmp_path, out_dir=tmp_path / "tuned")
        assert got["runner_up_value"] == pytest.approx(0.140)

    def test_a_sweep_with_no_completed_trials_returns_none(self, tmp_path):
        """Rather than inventing a winner from an empty directory."""
        from iqssl.cli.sweep import save_best_params

        assert save_best_params("vicreg", sweep_root=tmp_path, out_dir=tmp_path / "t") is None
