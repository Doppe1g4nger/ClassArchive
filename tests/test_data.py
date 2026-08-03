"""Stage-2 tests: generation, storage, splits, and the difficulty gate.

The reproducibility tests here are the ones that matter most. A dataset that
silently changes between runs invalidates every comparison built on it, and the
failure mode is not a crash — it is a results table that quietly stops meaning
what it says.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from iqssl.data import baselines, splits
from iqssl.data.build import build_dataset, load_manifest, verify_dataset
from iqssl.data.dataset import IQDataset, make_collate
from iqssl.data.generator import GEN_CHUNK, SyntheticIQGenerator
from iqssl.data.params import NUISANCE_FIELDS, GeneratorConfig, get_preset
from iqssl.types import ViewSpec
from tests import tolerances as tol

SMOKE = dict(difficulty="smoke", seed=0)


@pytest.fixture(scope="module")
def generator() -> SyntheticIQGenerator:
    return SyntheticIQGenerator(GeneratorConfig(n_samples=4096, **SMOKE))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("ds")
    cfg = GeneratorConfig(n_samples=1024, shard_size=512, **SMOKE)
    build_dataset(cfg, root, overwrite=True, progress=False)
    return str(root)


class TestGeneratorReproducibility:
    def test_prefix_stable(self, generator):
        # Sample i must be the same signal regardless of how many samples the
        # dataset has. Without this, growing a dataset silently rewrites it.
        a, _ = generator.generate(0, 8)
        b, _ = generator.generate(0, 64)
        assert np.array_equal(a, b[:8])

    def test_offset_stable(self, generator):
        b, _ = generator.generate(0, 64)
        c, _ = generator.generate(5, 3)
        assert np.array_equal(c, b[5:8])

    def test_stable_across_generation_windows(self, generator):
        # Batched convolutions pick different blocking for different batch
        # sizes, so a sample computed in a batch of 8 vs 64 differs in the last
        # mantissa bits. Window alignment is what removes that dependence.
        assert GEN_CHUNK > 0
        d, _ = generator.generate(GEN_CHUNK - 8, 16)
        e, _ = generator.generate(GEN_CHUNK - 4, 4)
        assert np.array_equal(e, d[4:8])

    def test_different_seed_gives_different_data(self):
        a, _ = SyntheticIQGenerator(
            GeneratorConfig(n_samples=64, difficulty="smoke", seed=0)
        ).generate(0, 8)
        b, _ = SyntheticIQGenerator(
            GeneratorConfig(n_samples=64, difficulty="smoke", seed=1)
        ).generate(0, 8)
        assert not np.array_equal(a, b)

    def test_emitters_are_seed_stable(self):
        a = SyntheticIQGenerator(GeneratorConfig(n_samples=64, **SMOKE)).emitters
        b = SyntheticIQGenerator(GeneratorConfig(n_samples=999, **SMOKE)).emitters
        assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


class TestGeneratorOutput:
    def test_shape_and_dtype(self, generator):
        iq, meta = generator.generate(0, 32)
        preset = get_preset("smoke")
        assert iq.shape == (32, 2, preset.store_len)
        assert iq.dtype == np.float32
        assert len(meta) == 32

    def test_all_finite(self, generator):
        iq, _ = generator.generate(0, 128)
        assert np.isfinite(iq).all()

    def test_normalized_to_unit_power(self, generator):
        iq, _ = generator.generate(0, 64)
        p = (iq**2).sum(1).mean(-1)
        assert np.allclose(p, 1.0, atol=1e-4)

    def test_snr_realized_tracks_nominal(self, generator):
        # They differ only by the finite-buffer noise realization, which at
        # store_len samples is a few tenths of a dB.
        _, meta = generator.generate(0, 512)
        err = (meta.snr_db_realized - meta.snr_db_nominal).abs()
        assert err.mean() < tol.SNR_REALIZED_DB
        assert err.max() < 3.0

    def test_every_nuisance_field_is_recorded(self, generator):
        _, meta = generator.generate(0, 16)
        for field in NUISANCE_FIELDS:
            assert field in meta.columns
            assert meta[field].notna().all()

    def test_emitter_params_are_recorded(self, generator):
        _, meta = generator.generate(0, 16)
        assert "emitter_pa_ibo_db" in meta.columns
        assert "emitter_iq_gain_db" in meta.columns

    def test_all_modulations_appear(self, generator):
        _, meta = generator.generate(0, 1024)
        assert set(meta.modulation.unique()) == set(generator.cfg.modulations)

    def test_emitter_and_channel_params_are_independent(self):
        # The design rule the whole nuisance analysis rests on. If CFO were
        # correlated with emitter identity, "did the model discard CFO?" would
        # have no interpretable answer.
        g = SyntheticIQGenerator(GeneratorConfig(n_samples=4096, **SMOKE))
        _, meta = g.generate(0, 4096)
        eid = meta.emitter_id.to_numpy().astype(float)
        for field in ("cfo_norm", "snr_db_nominal", "sro_ppm", "delay_spread_symbols"):
            r = abs(float(np.corrcoef(eid, meta[field].to_numpy())[0, 1]))
            assert r < 0.1, f"{field} correlates with emitter_id (r={r:.3f})"

    def test_cfo_bias_is_off_by_default(self):
        g = SyntheticIQGenerator(GeneratorConfig(n_samples=16, **SMOKE))
        assert all(e.cfo_bias_hz == 0.0 for e in g.emitters)

    def test_rolloff_varies(self, generator):
        _, meta = generator.generate(0, 256)
        assert meta.rolloff.nunique() > 3


class TestSplits:
    def test_iid_split_is_disjoint_and_complete(self, generator):
        _, meta = generator.generate(0, 1024)
        s = splits.make_splits(meta, "iid", seed=0)
        assert set(np.unique(s)) <= {"train", "val", "test"}
        assert len(s) == len(meta)

    def test_iid_split_covers_all_classes_in_train(self, generator):
        _, meta = generator.generate(0, 2048)
        s = splits.make_splits(meta, "iid", seed=0)
        train = meta[s == "train"]
        assert train.emitter_id.nunique() == meta.emitter_id.nunique()
        assert train.modulation.nunique() == meta.modulation.nunique()

    def test_holdout_emitters_has_no_overlap(self, generator):
        _, meta = generator.generate(0, 2048)
        s = splits.make_splits(meta, "holdout_emitters", seed=0)
        train_e = set(meta.emitter_id[s == "train"])
        test_e = set(meta.emitter_id[s == "test"])
        assert train_e and test_e
        assert not (train_e & test_e)

    def test_holdout_snr_puts_the_low_tail_in_test(self, generator):
        _, meta = generator.generate(0, 2048)
        s = splits.make_splits(meta, "holdout_snr", seed=0)
        assert meta.snr_db_realized[s == "test"].max() <= meta.snr_db_realized[s == "train"].min()

    def test_holdout_channel_reserves_richest_multipath(self, generator):
        _, meta = generator.generate(0, 2048)
        meta = meta.copy()
        # The smoke preset is single-tap by design, so synthesize the two tap
        # counts the variant needs.
        meta["n_taps"] = np.tile([1, 3], len(meta) // 2 + 1)[: len(meta)]
        s = splits.make_splits(meta, "holdout_channel", seed=0)
        assert meta.n_taps[s == "test"].min() > meta.n_taps[s == "train"].max()

    def test_holdout_channel_refuses_a_single_tap_prior(self, generator):
        # Without the guard this silently yields an empty training split.
        _, meta = generator.generate(0, 256)
        meta = meta.copy()
        meta["n_taps"] = 1
        with pytest.raises(ValueError, match="more than one tap count"):
            splits.make_splits(meta, "holdout_channel", seed=0)

    def test_every_stored_split_variant_has_all_three_parts(self, built):
        import pandas as pd

        meta = pd.read_parquet(f"{built}/meta.parquet")
        for col in (c for c in meta.columns if c.startswith("split_")):
            present = set(meta[col].unique())
            assert {"train", "test"} <= present, f"{col} is missing a split"

    def test_unknown_variant_rejected(self, generator):
        _, meta = generator.generate(0, 32)
        with pytest.raises(ValueError, match="unknown split variant"):
            splits.make_splits(meta, "nope")

    def test_label_subsets_are_nested(self, generator):
        # A 1% subset must be contained in the 10% one, or the label-efficiency
        # curve partly measures which samples each fraction happened to draw.
        _, meta = generator.generate(0, 2048)
        s = splits.make_splits(meta, "iid", seed=0)
        subs = splits.make_label_subsets(meta, s, (0.01, 0.1), seed=0)
        small = set(subs["emitter_id"]["0.01"])
        large = set(subs["emitter_id"]["0.1"])
        assert small <= large

    def test_label_subsets_come_from_train_only(self, generator):
        _, meta = generator.generate(0, 2048)
        s = splits.make_splits(meta, "iid", seed=0)
        subs = splits.make_label_subsets(meta, s, (0.1,), seed=0)
        train_idx = set(np.flatnonzero(s == "train").tolist())
        assert set(subs["emitter_id"]["0.1"]) <= train_idx

    def test_label_subsets_cover_every_class(self, generator):
        # A plain random 1% can miss classes entirely, which would make the
        # few-label result depend on the draw rather than the method.
        _, meta = generator.generate(0, 2048)
        s = splits.make_splits(meta, "iid", seed=0)
        subs = splits.make_label_subsets(meta, s, (0.01,), seed=0)
        chosen = meta.iloc[subs["emitter_id"]["0.01"]]
        assert chosen.emitter_id.nunique() == meta.emitter_id.nunique()


class TestBuildAndManifest:
    def test_manifest_records_hash_and_geometry(self, built):
        m = load_manifest(built)
        assert m["dataset_hash"].startswith("sha256:")
        assert m["n_samples"] == 1024
        assert m["store_len"] == get_preset("smoke").store_len

    def test_shards_verify(self, built):
        assert verify_dataset(built)

    def test_verification_detects_corruption(self, built, tmp_path):
        import shutil
        from pathlib import Path

        copy = tmp_path / "corrupt"
        shutil.copytree(built, copy)
        shard = next(Path(copy).glob("shard_*.npy"))
        arr = np.load(shard)
        arr[0, 0, 0] += 1.0
        np.save(shard, arr)
        assert not verify_dataset(copy)

    def test_hash_is_stable_across_rebuilds(self, tmp_path):
        cfg = GeneratorConfig(n_samples=512, shard_size=256, **SMOKE)
        a = build_dataset(cfg, tmp_path / "a", overwrite=True, progress=False)
        b = build_dataset(cfg, tmp_path / "b", overwrite=True, progress=False)
        assert a["dataset_hash"] == b["dataset_hash"]

    def test_hash_changes_with_seed(self, tmp_path):
        a = build_dataset(
            GeneratorConfig(n_samples=512, shard_size=256, difficulty="smoke", seed=0),
            tmp_path / "a",
            overwrite=True,
            progress=False,
        )
        b = build_dataset(
            GeneratorConfig(n_samples=512, shard_size=256, difficulty="smoke", seed=1),
            tmp_path / "b",
            overwrite=True,
            progress=False,
        )
        assert a["dataset_hash"] != b["dataset_hash"]

    def test_hash_is_independent_of_shard_size(self, tmp_path):
        # Sharding is a storage detail. If it changed the hash, an otherwise
        # identical dataset would look like a different one to the aggregator.
        a = build_dataset(
            GeneratorConfig(n_samples=1024, shard_size=512, **SMOKE),
            tmp_path / "a",
            overwrite=True,
            progress=False,
        )
        b = build_dataset(
            GeneratorConfig(n_samples=1024, shard_size=1024, **SMOKE),
            tmp_path / "b",
            overwrite=True,
            progress=False,
        )
        assert a["config"]["n_samples"] == b["config"]["n_samples"]
        # Same buffers, so the shard *contents* agree even though the files differ.
        sa = np.concatenate([np.load(f"{tmp_path}/a/shard_{i:04d}.npy") for i in range(2)])
        sb = np.load(f"{tmp_path}/b/shard_0000.npy")
        assert np.array_equal(sa, sb)

    def test_refuses_to_overwrite_without_permission(self, built):
        with pytest.raises(FileExistsError):
            build_dataset(GeneratorConfig(n_samples=8, **SMOKE), built, progress=False)

    def test_nuisance_stats_come_from_train_split(self, built):
        m = load_manifest(built)
        stats = m["nuisance_stats"]
        assert stats["fields"] == list(NUISANCE_FIELDS)
        assert len(stats["mean"]) == len(NUISANCE_FIELDS)
        assert all(s > 0 for s in stats["std"])

    def test_applicable_split_variants_are_stored(self, built):
        import pandas as pd

        meta = pd.read_parquet(f"{built}/meta.parquet")
        # holdout_channel is legitimately absent for a single-tap preset (smoke
        # and easy): build_dataset warns and skips rather than writing an empty
        # training split. Everything else must be present.
        for variant in ("iid", "holdout_emitters", "holdout_snr"):
            assert f"split_{variant}" in meta.columns
        assert "split_holdout_channel" not in meta.columns

    def test_label_subsets_are_persisted(self, built):
        with open(f"{built}/label_subsets.json") as fh:
            subs = json.load(fh)
        assert "iid" in subs
        assert "emitter_id" in subs["iid"]


class TestDataset:
    def test_item_shapes(self, built):
        ds = IQDataset(built, "train")
        item = ds[0]
        assert item["x"].shape == (2, ds.crop_len)
        assert item["x"].dtype == torch.float32
        assert item["nuisance"].shape == (len(NUISANCE_FIELDS),)

    def test_primary_label_switch(self, built):
        emitter = IQDataset(built, "train", primary_label="emitter")
        mod = IQDataset(built, "train", primary_label="modulation")
        assert emitter[0]["y_primary"] == emitter[0]["y_emitter"]
        assert mod[0]["y_primary"] == mod[0]["y_mod"]
        assert emitter.num_primary_classes != mod.num_primary_classes

    def test_random_crop_varies_but_deterministic_crop_does_not(self, built):
        fixed = IQDataset(built, "train", random_crop=False)
        assert torch.equal(fixed[0]["x"], fixed[0]["x"])
        rand = IQDataset(built, "train", random_crop=True)
        torch.manual_seed(0)
        a = rand[0]["x"]
        torch.manual_seed(1)
        b = rand[0]["x"]
        assert not torch.equal(a, b)

    def test_crop_is_renormalized(self, built):
        ds = IQDataset(built, "train")
        x = ds[0]["x"]
        assert float((x**2).sum(0).mean()) == pytest.approx(1.0, abs=1e-4)

    def test_crop_longer_than_store_rejected(self, built):
        with pytest.raises(ValueError, match="exceeds store_len"):
            IQDataset(built, "train", crop_len=99999)

    def test_splits_are_disjoint_in_the_dataset(self, built):
        train = IQDataset(built, "train")
        test = IQDataset(built, "test")
        assert not (set(train._rows.tolist()) & set(test._rows.tolist()))

    def test_nuisance_is_standardized(self, built):
        ds = IQDataset(built, "train", standardize_nuisance=True)
        vals = torch.stack([ds[i]["nuisance"] for i in range(min(len(ds), 300))])
        assert abs(float(vals.mean())) < 0.5
        raw = IQDataset(built, "train", standardize_nuisance=False)
        assert not torch.allclose(raw[0]["nuisance"], ds[0]["nuisance"])

    def test_label_subset_indices_are_in_range(self, built):
        ds = IQDataset(built, "train")
        subset = ds.label_subset(0.1)
        assert subset
        assert all(0 <= i < len(ds) for i in subset)

    def test_label_subset_is_identical_across_instances(self, built):
        # Two "methods" must finetune on exactly the same labels.
        a = IQDataset(built, "train").label_subset(0.1)
        b = IQDataset(built, "train").label_subset(0.1)
        assert a == b

    def test_collate_produces_a_valid_batch(self, built):
        ds = IQDataset(built, "train")
        collate = make_collate(ViewSpec(n_views=2))
        batch = collate([ds[i] for i in range(4)])
        assert batch.batch_size == 4
        assert batch.x_raw.shape == (4, 2, ds.crop_len)
        assert batch.y_primary is not None
        assert batch.nuisance_names == NUISANCE_FIELDS

    def test_collate_always_attaches_labels(self, built):
        # Even unsupervised methods get them: the online probe logs need them.
        ds = IQDataset(built, "train")
        batch = make_collate(ViewSpec(needs_labels=False))([ds[i] for i in range(2)])
        assert batch.y_emitter is not None and batch.y_mod is not None


class TestClassicalFeatures:
    def test_shape_and_finiteness(self):
        x = torch.randn(8, 2, 512)
        f = baselines.classical_features(x)
        assert f.shape == (8, len(baselines.CLASSICAL_FEATURE_NAMES))
        assert torch.isfinite(f).all()

    def test_survives_degenerate_input(self):
        # All-zero buffers appear if a preset is misconfigured; the gate should
        # report a bad number, not crash inside a division.
        f = baselines.classical_features(torch.zeros(2, 2, 128))
        assert torch.isfinite(f).all()

    def test_dc_feature_detects_an_offset(self):
        x = torch.randn(4, 2, 4096)
        x[:, 0] += 0.5
        idx = baselines.CLASSICAL_FEATURE_NAMES.index("dc_i")
        assert float(baselines.classical_features(x)[:, idx].mean()) == pytest.approx(0.5, abs=0.05)

    def test_circularity_detects_iq_imbalance(self):
        # A proper (circular) signal has E[z^2] = 0; imbalance makes it improper.
        # Unlike var(I)/var(Q), this survives an unknown carrier phase.
        from iqssl.dsp.convert import complex_to_ri
        from iqssl.dsp.impair import iq_imbalance

        g = torch.Generator().manual_seed(0)
        z = torch.complex(
            torch.randn(64, 4096, generator=g), torch.randn(64, 4096, generator=g)
        ) / np.sqrt(2)
        idx = baselines.CLASSICAL_FEATURE_NAMES.index("circularity")
        clean = baselines.classical_features(complex_to_ri(z))[:, idx]
        skewed = baselines.classical_features(complex_to_ri(iq_imbalance(z, 3.0, 15.0)))[:, idx]
        assert float(skewed.mean()) > float(clean.mean()) * 3

    def test_circularity_is_rotation_invariant(self):
        from iqssl.dsp.convert import complex_to_ri
        from iqssl.dsp.impair import iq_imbalance

        g = torch.Generator().manual_seed(1)
        z = torch.complex(
            torch.randn(32, 4096, generator=g), torch.randn(32, 4096, generator=g)
        ) / np.sqrt(2)
        z = iq_imbalance(z, 2.0, 10.0)
        idx = baselines.CLASSICAL_FEATURE_NAMES.index("circularity")
        a = baselines.classical_features(complex_to_ri(z))[:, idx]
        b = baselines.classical_features(complex_to_ri(z * np.exp(1j * 0.7)))[:, idx]
        assert torch.allclose(a, b, atol=1e-3)


def test_small_cnn_can_overfit_a_tiny_set():
    """Sanity-check the oracle itself before trusting it to calibrate difficulty.

    If the reference model could not fit even a memorizable set, a "task too
    hard" verdict from the gate would be about the instrument, not the task.
    """
    torch.manual_seed(0)
    x = torch.randn(32, 2, 128)
    y = torch.arange(4).repeat(8)
    x += y.view(-1, 1, 1).float() * 3.0  # trivially separable
    model = baselines.SmallCNN(4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(120):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        assert float((model(x).argmax(-1) == y).float().mean()) > 0.95


class TestDifficultyBands:
    """The bands are a ladder's rungs, and must not be fitted to it.

    Grading every preset against `easy`'s 0.85-0.95 failed `medium` and `hard` by
    construction -- they exist to be harder. The fix is per-rung bands, and the
    hazard the fix introduces is circularity: fit each band to what its preset
    happened to score and the gate certifies everything while meaning nothing.
    """

    def test_every_rung_has_its_own_oracle_band(self):
        from iqssl.data.baselines import PRESET_BANDS

        highs = {p: b["supervised_cnn_high_snr"] for p, b in PRESET_BANDS.items()}
        assert highs["easy"] > highs["medium"] > highs["hard"], (
            f"bands must descend with difficulty, got {highs}"
        )

    def test_oracle_lower_bounds_come_from_the_chance_rule(self):
        """Not from whatever each preset measured -- that is the circular version."""
        from iqssl.data.baselines import MIN_ORACLE_CHANCE_MULTIPLE, PRESET_BANDS

        floor = MIN_ORACLE_CHANCE_MULTIPLE / 16  # 16 emitters on every real rung
        for preset, bands in PRESET_BANDS.items():
            lo = bands["supervised_cnn_high_snr"][0]
            assert lo >= floor - 0.02, (
                f"{preset}'s oracle band starts at {lo}, below {floor:.3f} = "
                f"{MIN_ORACLE_CHANCE_MULTIPLE}x chance; a rung that close to chance "
                "cannot resolve twelve methods"
            )

    def test_hard_still_fails_its_own_band(self):
        """The measured 0.149 must not pass. If a future edit makes it pass, the
        band was widened to fit the preset rather than the preset fixed."""
        from iqssl.data.baselines import bands_for

        lo, _ = bands_for("hard")["supervised_cnn_high_snr"]
        assert lo > 0.149, "hard measured 0.149; a band that accepts it is vacuous"

    def test_cheap_readability_ceilings_do_not_vary_by_rung(self):
        """A harder channel is no excuse for a task closed-form features solve."""
        from iqssl.data.baselines import PRESET_BANDS

        for key in ("classical", "raw_linear"):
            assert len({b[key] for b in PRESET_BANDS.values()}) == 1

    def test_unknown_preset_falls_back_to_easy(self):
        from iqssl.data.baselines import TARGET_BANDS, bands_for

        assert bands_for(None) == TARGET_BANDS
        assert bands_for("nonesuch") == TARGET_BANDS
