"""Stage-8 tests: the tuning objective and the equal budget.

Both properties here are the kind that never fail loudly. A sweep that selects
on the test split still produces a plausible table; so does one that gives a
method thirty trials. Neither is detectable after the fact from the numbers, so
they are asserted before the compute is spent.
"""

from __future__ import annotations

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

        cmd = " ".join(build_command("simclr", "hpo", N_TRIALS, []))
        assert f"hydra.sweeper.n_trials={N_TRIALS}" in cmd
        assert "hydra/sweeper=equal_budget" in cmd
        assert "hydra.sweeper.params.method.args.temperature" in cmd

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
