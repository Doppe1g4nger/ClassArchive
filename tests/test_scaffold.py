"""Stage-0 tests: the contracts everything else is built on."""

from __future__ import annotations

import pytest
import torch

from iqssl.registry import Registry
from iqssl.types import Batch, MethodOutput, ViewSpec
from iqssl.utils import hashing, seed


class TestRegistry:
    def test_register_and_get(self):
        reg: Registry = Registry("widgets")

        @reg.register("foo")
        class Foo:
            pass

        assert reg.get("foo") is Foo
        assert reg.keys() == ["foo"]
        assert "foo" in reg

    def test_default_key_is_lowercased_class_name(self):
        reg: Registry = Registry("widgets")

        @reg.register()
        class Bar:
            pass

        assert reg.get("bar") is Bar

    def test_unknown_key_lists_available(self):
        reg: Registry = Registry("widgets")

        @reg.register("foo")
        class Foo:
            pass

        with pytest.raises(KeyError, match=r"available.*foo"):
            reg.get("nope")

    def test_duplicate_key_rejected(self):
        reg: Registry = Registry("widgets")

        @reg.register("foo")
        class Foo:
            pass

        with pytest.raises(ValueError, match="duplicate key"):

            @reg.register("foo")
            class Other:
                pass

    def test_reregistering_same_class_is_idempotent(self):
        # autodiscover() may import a module twice via different paths.
        reg: Registry = Registry("widgets")

        class Foo:
            pass

        reg.register("foo")(Foo)
        reg.register("foo")(Foo)
        assert len(reg) == 1


class TestViewSpec:
    def test_defaults(self):
        spec = ViewSpec()
        assert spec.n_views == 2
        assert not spec.needs_mask

    def test_mask_kind_required_with_needs_mask(self):
        with pytest.raises(ValueError, match="explicit mask_kind"):
            ViewSpec(needs_mask=True)

    def test_mask_kind_without_needs_mask_rejected(self):
        with pytest.raises(ValueError, match="needs_mask=False"):
            ViewSpec(needs_mask=False, mask_kind="random")

    def test_negative_views_rejected(self):
        with pytest.raises(ValueError, match="n_views"):
            ViewSpec(n_views=-1)

    def test_is_hashable(self):
        # Frozen so it can key a cache of collate functions.
        assert hash(ViewSpec()) == hash(ViewSpec())


class TestBatch:
    def _batch(self, b=4, ell=64):
        return Batch(
            x_raw=torch.randn(b, 2, ell),
            views=[torch.randn(b, 2, ell), torch.randn(b, 2, ell)],
            y_primary=torch.arange(b),
        )

    def test_shape_properties(self):
        batch = self._batch()
        assert batch.batch_size == 4
        assert batch.seq_len == 64

    def test_to_device_moves_all_fields(self):
        batch = self._batch()
        moved = batch.to("cpu")
        assert moved.x_raw.device.type == "cpu"
        assert all(v.device.type == "cpu" for v in moved.views)
        assert moved.y_primary is not None

    def test_require_views_error_names_caller(self):
        batch = Batch(x_raw=torch.randn(2, 2, 8), views=[torch.randn(2, 2, 8)])
        with pytest.raises(ValueError, match="SimCLR needs 2 views but the batch has 1"):
            batch.require_views(2, "SimCLR")

    def test_require_primary_error_is_actionable(self):
        batch = Batch(x_raw=torch.randn(2, 2, 8))
        with pytest.raises(ValueError, match="needs_labels"):
            batch.require_primary("SupCon")


class TestMethodOutput:
    def test_rejects_non_scalar_loss(self):
        with pytest.raises(ValueError, match="scalar"):
            MethodOutput(loss=torch.zeros(3))

    def test_accepts_scalar(self):
        out = MethodOutput(loss=torch.tensor(1.0), logs={"loss": 1.0})
        assert out.logs["loss"] == 1.0


class TestSeeding:
    def test_seed_everything_is_reproducible(self):
        seed.seed_everything(123)
        a = torch.randn(10)
        seed.seed_everything(123)
        assert torch.equal(a, torch.randn(10))

    def test_spawn_streams_are_independent_and_reproducible(self):
        s1 = seed.spawn_streams(7, 3)
        s2 = seed.spawn_streams(7, 3)
        draws1 = [g.normal(size=5) for g in s1]
        draws2 = [g.normal(size=5) for g in s2]
        for a, b in zip(draws1, draws2, strict=True):
            assert (a == b).all()
        # Different streams must not coincide.
        assert not (draws1[0] == draws1[1]).all()

    def test_spawn_is_prefix_stable(self):
        # Generating N samples must not change what sample i is for i < N.
        # This is what makes the dataset regenerable at a different size.
        few = seed.spawn_keys(42, 4)
        many = seed.spawn_keys(42, 64)
        assert few == many[:4]

    def test_generator_from_key_round_trips(self):
        keys = seed.spawn_keys(11, 2)
        a = seed.generator_from_key(keys[0]).normal(size=8)
        b = seed.generator_from_key(keys[0]).normal(size=8)
        assert (a == b).all()

    def test_temp_seed_restores_state(self):
        seed.seed_everything(0)
        before = torch.randn(4)
        seed.seed_everything(0)
        with seed.temp_seed(999):
            torch.randn(100)
        assert torch.equal(before, torch.randn(4))


class TestHashing:
    def test_canonical_json_is_key_order_invariant(self):
        assert hashing.canonical_json({"a": 1, "b": 2}) == hashing.canonical_json({"b": 2, "a": 1})

    def test_config_hash_detects_value_change(self):
        h1 = hashing.hash_config({"lr": 0.1, "wd": 1e-4})
        h2 = hashing.hash_config({"lr": 0.2, "wd": 1e-4})
        assert h1 != h2

    def test_dataset_hash_includes_generator_version(self):
        shards = {"shard_0000.npy": "a" * 64}
        cfg = {"n": 10}
        assert hashing.dataset_hash(shards, cfg, "v1") != hashing.dataset_hash(shards, cfg, "v2")

    def test_dataset_hash_is_shard_order_invariant(self):
        cfg = {"n": 10}
        a = hashing.dataset_hash({"s0": "x" * 64, "s1": "y" * 64}, cfg, "v1")
        b = hashing.dataset_hash({"s1": "y" * 64, "s0": "x" * 64}, cfg, "v1")
        assert a == b

    def test_short_strips_algorithm_prefix(self):
        assert hashing.short("sha256:abcdef0123456789") == "abcdef012345"


def test_csv_logger_handles_keys_appearing_mid_run(tmp_path):
    import csv as _csv

    from iqssl.utils.logging_ import CSVLogger

    logger = CSVLogger(tmp_path / "metrics.csv")
    logger.log({"loss": 1.0}, step=0)
    logger.log({"loss": 0.5, "probe_acc": 0.3}, step=1)
    logger.close()

    with open(tmp_path / "metrics.csv") as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 2
    assert "probe_acc" in rows[0]
    assert rows[0]["probe_acc"] == ""
    assert float(rows[1]["probe_acc"]) == 0.3
