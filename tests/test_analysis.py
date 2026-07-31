"""Stage-9 tests: collection, pooling refusal, tables, figures.

Aggregation is where every upstream guarantee can still be undone. A table
averaged over two dataset versions, two encoder widths or two augmentation
policies looks exactly like a correct one, and nothing downstream of it can
tell. So the refusals are tested at least as carefully as the arithmetic.

Fixtures are hand-written rather than generated: a test whose expected answer
comes from the same code that produced it verifies only self-consistency.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from iqssl.analysis.collect import collect_runs
from iqssl.analysis.figures import pareto_front
from iqssl.analysis.invariants import (
    MAY_DIFFER,
    MUST_AGREE,
    IncomparableRuns,
    check_invariants,
)
from iqssl.analysis.tables import headline_table, nuisance_table, to_markdown


def _meta(run: str, **overrides) -> dict:
    base = {
        "run": run,
        "method": "simclr",
        "dataset_hash": "sha256:abc",
        "seed": 0,
        "epochs": 10,
        "batch_size": 256,
        "warmup_frac": 0.1,
        "grad_clip": 1.0,
        "augment": "standard",
        "crop_len": 1024,
        "split_variant": "iid",
        "encoder": {"name": "vit1d", "args": {"size": "small"}},
        "optimizer": "lars",
        "base_lr": 0.3,
        "weight_decay": 1e-6,
    }
    return {**base, **overrides}


class TestInvariants:
    def test_identical_runs_pool(self):
        assert check_invariants([_meta("a"), _meta("b")]) == []

    def test_a_single_run_is_always_poolable(self):
        assert check_invariants([_meta("a", dataset_hash="anything")]) == []

    @pytest.mark.parametrize(
        ("key", "other"),
        [
            ("dataset_hash", "sha256:def"),
            ("epochs", 200),
            ("batch_size", 64),
            ("augment", "hardware_invariant"),
            ("crop_len", 4096),
            ("split_variant", "holdout_emitters"),
            ("grad_clip", 0.0),
            ("warmup_frac", 0.5),
        ],
    )
    def test_disagreement_on_a_held_constant_field_is_refused(self, key, other):
        with pytest.raises(IncomparableRuns, match=key):
            check_invariants([_meta("a"), _meta("b", **{key: other})])

    def test_encoder_differences_are_refused_by_value(self):
        """Nested config compared structurally: two dicts that differ must be
        caught, and two that merely aren't the same object must not be."""
        with pytest.raises(IncomparableRuns, match="encoder"):
            check_invariants(
                [_meta("a"), _meta("b", encoder={"name": "vit1d", "args": {"size": "base"}})]
            )
        assert (
            check_invariants(
                [_meta("a"), _meta("b", encoder={"name": "vit1d", "args": {"size": "small"}})]
            )
            == []
        )

    @pytest.mark.parametrize("key", MAY_DIFFER)
    def test_deliberately_varied_fields_still_pool(self, key):
        """method and seed are the axes being compared; optimizer, lr and weight
        decay vary by contract. Refusing these would make the benchmark unable
        to aggregate the very thing it exists to aggregate."""
        assert check_invariants([_meta("a"), _meta("b", **{key: "different"})]) == []

    def test_a_field_present_in_some_runs_only_is_refused(self):
        """The case where a default gets silently substituted for a real value."""
        partial = _meta("b")
        del partial["augment"]
        with pytest.raises(IncomparableRuns, match="augment"):
            check_invariants([_meta("a"), partial])

    def test_a_field_absent_everywhere_is_not_a_violation(self):
        # Older runs predate some keys; that is not a comparability problem.
        a, b = _meta("a"), _meta("b")
        del a["crop_len"], b["crop_len"]
        assert check_invariants([a, b]) == []

    def test_violations_can_be_reported_without_raising(self):
        found = check_invariants([_meta("a"), _meta("b", epochs=99)], raise_on_violation=False)
        assert len(found) == 1
        assert found[0].key == "epochs"
        assert "99" in found[0].describe()

    def test_the_two_lists_do_not_overlap(self):
        """A field in both would make the contract self-contradictory."""
        assert not set(MUST_AGREE) & set(MAY_DIFFER)


class TestTables:
    def _scores(self, rows) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "run": f"r{i}",
                    "method": m,
                    "seed": s,
                    "axis": "emitter",
                    "probe": "linear_probe",
                    "fraction": 1.0,
                    "score": v,
                }
                for i, (m, s, v) in enumerate(rows)
            ]
        )

    def test_seed_mean_and_sample_std(self):
        table = headline_table(self._scores([("simclr", 0, 0.4), ("simclr", 1, 0.6)]))
        row = table.iloc[0]
        assert row["mean"] == pytest.approx(0.5)
        # Sample std (ddof=1) of {0.4, 0.6} is 0.1414, not the population 0.1.
        assert row["std"] == pytest.approx(np.std([0.4, 0.6], ddof=1))
        assert row["n_seeds"] == 2

    def test_single_seed_reports_no_spread_rather_than_zero(self):
        """A zero error bar on one seed claims perfect reproducibility that
        nobody measured, and a reader takes it at face value."""
        table = headline_table(self._scores([("simclr", 0, 0.4)]))
        assert np.isnan(table.iloc[0]["std"])
        assert "±" not in to_markdown(table, "t")
        assert "not results" in to_markdown(table, "t")

    def test_methods_are_kept_separate(self):
        table = headline_table(self._scores([("simclr", 0, 0.4), ("byol", 0, 0.9)]))
        assert set(table["method"]) == {"simclr", "byol"}
        assert table.set_index("method").loc["byol", "mean"] == pytest.approx(0.9)

    def test_empty_input_gives_an_empty_table_not_an_error(self):
        empty = pd.DataFrame(columns=["axis", "probe", "fraction", "method", "seed", "score"])
        assert headline_table(empty).empty
        assert "(no runs)" in to_markdown(headline_table(empty), "t")

    def test_nuisance_table_keys_on_field(self):
        scores = pd.DataFrame(
            [
                {
                    "run": "r0",
                    "method": "simclr",
                    "seed": 0,
                    "axis": "nuisance",
                    "probe": "cfo_norm",
                    "fraction": float("nan"),
                    "score": 0.05,
                }
            ]
        )
        table = nuisance_table(scores)
        assert table.iloc[0]["nuisance_field"] == "cfo_norm"


class TestPareto:
    def test_frontier_keeps_only_undominated_points(self):
        # (cost, score): B is both cheaper and better than C, so C is dominated.
        cost = np.array([1.0, 2.0, 3.0])
        score = np.array([0.1, 0.5, 0.4])
        assert sorted(pareto_front(cost, score).tolist()) == [0, 1]

    def test_monotone_improvement_keeps_everything(self):
        cost = np.array([1.0, 2.0, 3.0])
        score = np.array([0.1, 0.2, 0.3])
        assert sorted(pareto_front(cost, score).tolist()) == [0, 1, 2]

    def test_a_cheaper_better_point_dominates_everything_after(self):
        cost = np.array([1.0, 5.0, 9.0])
        score = np.array([0.9, 0.5, 0.4])
        assert pareto_front(cost, score).tolist() == [0]


class TestCollect:
    def _write_run(self, root, method: str, seed: int, score: float):
        run = root / method / f"seed{seed}" / "ts"
        run.mkdir(parents=True)
        (run / "config.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "dataset_hash": "sha256:abc",
                    "train": {"seed": seed, "epochs": 10, "batch_size": 32, "augment": "light"},
                }
            )
        )
        (run / "summary.json").write_text(json.dumps({"compute/tokens_seen": 1000.0}))
        (run / "eval.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "split_variant": "iid",
                    "finetune_included": True,
                    "axes": {
                        "emitter": {
                            "linear_probe": {"1": score},
                            "knn": {"1": score},
                            "finetune": {},
                            "snr_quartiles": [
                                {"snr_lo": 0.0, "snr_hi": 5.0, "n": 10, "accuracy": score}
                            ],
                        }
                    },
                    "nuisance_r2": {"cfo_norm": 0.02},
                }
            )
        )
        return run

    def test_collects_scores_and_metadata(self, tmp_path):
        self._write_run(tmp_path, "simclr", 0, 0.4)
        self._write_run(tmp_path, "simclr", 1, 0.6)
        scores, metas = collect_runs(tmp_path)

        assert len(metas) == 2
        probe = scores[(scores["probe"] == "linear_probe")]
        assert sorted(probe["score"]) == [0.4, 0.6]
        assert (scores["probe"] == "snr_band").any()
        assert (scores["axis"] == "nuisance").any()
        assert metas[0]["compute/tokens_seen"] == 1000.0

    def test_an_unevaluated_run_is_warned_about_not_silently_dropped(self, tmp_path, caplog):
        """Shrinking a seed group from three to two without saying so is how a
        crashed evaluation turns into a quietly weaker error bar."""
        import torch

        self._write_run(tmp_path, "simclr", 0, 0.4)
        orphan = tmp_path / "simclr" / "seed1" / "ts"
        orphan.mkdir(parents=True)
        torch.save({}, orphan / "checkpoint.pt")

        with caplog.at_level("WARNING"):
            _, metas = collect_runs(tmp_path)
        assert len(metas) == 1
        assert "no eval.json" in caplog.text

    def test_empty_root_returns_empty(self, tmp_path):
        scores, metas = collect_runs(tmp_path)
        assert scores.empty and metas == []
