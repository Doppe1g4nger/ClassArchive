"""Stage-7 tests: the evaluation protocol.

The probes are tested against constructed cases with known answers before the
protocol is trusted on real features: an evaluation bug does not crash, it
produces a plausible-looking number about nothing, and the results table
downstream cannot tell.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from iqssl.data.build import build_dataset
from iqssl.data.dataset import IQDataset
from iqssl.data.params import NUISANCE_FIELDS, GeneratorConfig
from iqssl.eval import probes
from iqssl.eval.loading import DatasetMismatch, check_dataset_hash
from iqssl.eval.protocol import evaluate_run
from iqssl.registry import autodiscover

autodiscover()


def _clusters(n_per: int = 60, n_classes: int = 4, dim: int = 16, gap: float = 6.0, seed: int = 0):
    """Well-separated Gaussian clusters: any sane classifier scores ~1.0.

    Centres come from a *fixed* seed and only the noise varies with ``seed``, so
    train and test sets drawn with different seeds share geometry — the first
    version of this fixture reseeded the centres too, and the probes were being
    asked to classify points around centres they had never seen.
    """
    centres = torch.randn(n_classes, dim, generator=torch.Generator().manual_seed(1234)) * gap
    g = torch.Generator().manual_seed(seed)
    z = torch.cat([centres[c] + torch.randn(n_per, dim, generator=g) for c in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per)
    return z, y


class TestKNN:
    def test_separates_clean_clusters(self):
        ztr, ytr = _clusters(seed=0)
        zte, yte = _clusters(seed=1)
        assert probes.knn_probe(ztr, ytr, zte, yte, 4) > 0.95

    def test_chance_on_pure_noise(self):
        g = torch.Generator().manual_seed(0)
        ztr, zte = torch.randn(400, 16, generator=g), torch.randn(200, 16, generator=g)
        ytr = np.random.default_rng(0).integers(0, 4, 400)
        yte = np.random.default_rng(1).integers(0, 4, 200)
        acc = probes.knn_probe(ztr, ytr, zte, yte, 4)
        assert 0.05 < acc < 0.45  # around chance (0.25), not degenerate

    def test_handles_fewer_train_samples_than_k(self):
        ztr, ytr = _clusters(n_per=3)  # 12 < k=20
        zte, yte = _clusters(n_per=5, seed=1)
        assert probes.knn_probe(ztr, ytr, zte, yte, 4) > 0.9


class TestLinearProbe:
    def test_separates_clean_clusters_and_returns_predictions(self):
        ztr, ytr = _clusters(seed=0)
        zte, yte = _clusters(seed=1)
        acc, pred = probes.linear_probe(ztr, ytr, zte, yte, 4)
        assert acc > 0.95
        assert pred.shape == yte.shape
        assert float((pred == yte).mean()) == pytest.approx(acc)

    def test_is_deterministic(self):
        # Two invocations must agree exactly, or method-to-method probe gaps
        # would carry probe-seed noise.
        ztr, ytr = _clusters(seed=0)
        zte, yte = _clusters(seed=1)
        a, _ = probes.linear_probe(ztr, ytr, zte, yte, 4)
        b, _ = probes.linear_probe(ztr, ytr, zte, yte, 4)
        assert a == b


class TestNuisanceRidge:
    def test_recovers_a_linearly_embedded_nuisance(self):
        """Features that contain the nuisance linearly must read R^2 ~ 1."""
        g = torch.Generator().manual_seed(0)
        w = torch.randn(16, 2, generator=g)
        ztr, zte = torch.randn(500, 16, generator=g), torch.randn(200, 16, generator=g)
        ntr, nte = (ztr @ w).numpy(), (zte @ w).numpy()
        r2 = probes.nuisance_r2(ztr, ntr, zte, nte)
        assert all(r > 0.99 for r in r2)

    def test_reads_zero_for_an_absent_nuisance(self):
        g = torch.Generator().manual_seed(0)
        ztr, zte = torch.randn(500, 16, generator=g), torch.randn(200, 16, generator=g)
        ntr = np.random.default_rng(0).normal(size=(500, 2))
        nte = np.random.default_rng(1).normal(size=(200, 2))
        r2 = probes.nuisance_r2(ztr, ntr, zte, nte)
        assert all(r < 0.15 for r in r2)


class TestFinetune:
    def test_learns_a_separable_task(self):
        """A trivially separable signal must be learned even from few epochs;
        if it is not, the finetune path is silently not training the encoder."""
        from iqssl.models.vit1d import vit1d

        torch.manual_seed(0)
        enc = vit1d(size="tiny", seq_len=64, patch_size=16, embed_dim=32, depth=1, num_heads=2)
        y = np.repeat(np.arange(2), 32)
        x = torch.randn(64, 2, 64) + torch.from_numpy(y).float().view(-1, 1, 1) * 3.0
        acc = probes.finetune_probe(enc, x, y, x, y, 2)
        assert acc > 0.9

    def test_does_not_mutate_the_frozen_encoder(self):
        """Each fraction must start from the pretrained point, not from the
        previous fraction's finetuned weights."""
        from iqssl.models.vit1d import vit1d

        torch.manual_seed(0)
        enc = vit1d(size="tiny", seq_len=64, patch_size=16, embed_dim=32, depth=1, num_heads=2)
        before = {k: v.clone() for k, v in enc.state_dict().items()}
        y = np.repeat(np.arange(2), 8)
        x = torch.randn(16, 2, 64)
        probes.finetune_probe(enc, x, y, x, y, 2)
        after = enc.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)


@pytest.fixture(scope="module")
def pretrained_run(tmp_path_factory):
    """A real (tiny) pretrained run: dataset, training, checkpoint, config."""
    from omegaconf import OmegaConf

    from iqssl.cli.pretrain import build_method, merge_train_config
    from iqssl.train.loop import train

    root = tmp_path_factory.mktemp("eval_ds")
    build_dataset(
        GeneratorConfig(n_samples=768, shard_size=768, difficulty="smoke", seed=0),
        root,
        overwrite=True,
        progress=False,
    )

    cfg = OmegaConf.create(
        {
            "experiment": "unittest",
            "output_root": str(tmp_path_factory.mktemp("runs")),
            "encoder": {"name": "vit1d", "args": {"size": "tiny", "patch_size": 16}},
            "method": {"name": "simclr", "args": {}},
            "data": {
                "root": str(root),
                "split_variant": "iid",
                "primary_label": "emitter",
                "crop_len": None,
            },
            "train": {
                "epochs": 1,
                "batch_size": 16,
                "max_steps": 3,
                "device": "cpu",
                "log_every": 1,
                "seed": 0,
                "augment": "light",
            },
        }
    )
    dataset = IQDataset(root, "train", random_crop=False)
    method = build_method(cfg, dataset.crop_len, seed=0, n_classes=dataset.num_primary_classes)

    run_dir = tmp_path_factory.mktemp("run") / "seed0"
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=run_dir / "config_full.yaml")
    train(method, dataset, merge_train_config(cfg), out_dir=run_dir, method_cfg=cfg)
    return run_dir, root


class TestProtocol:
    def test_end_to_end_writes_a_complete_report(self, pretrained_run):
        run_dir, root = pretrained_run
        report = evaluate_run(run_dir, root, skip_finetune=True)

        assert (run_dir / "eval.json").exists()
        assert report["finetune_included"] is False
        for axis in ("emitter", "modulation"):
            res = report["axes"][axis]
            for frac in ("0.01", "0.1", "1"):
                assert 0.0 <= res["linear_probe"][frac] <= 1.0
                assert 0.0 <= res["knn"][frac] <= 1.0
            assert len(res["snr_quartiles"]) == 4
        assert set(report["nuisance_r2"]) == set(NUISANCE_FIELDS)

    def test_finetune_column_present_when_not_skipped(self, pretrained_run):
        run_dir, root = pretrained_run
        report = evaluate_run(run_dir, root)
        assert report["finetune_included"] is True
        assert "1" in report["axes"]["emitter"]["finetune"]

    def test_a_partial_checkpoint_is_loadable_but_warns(self, pretrained_run, caplog):
        """Salvaging a killed run is the point; doing it unknowingly is not.

        A periodic checkpoint from a run that died is deliberately loadable --
        otherwise 90 minutes of compute is simply lost. But its scores are not
        comparable with a completed run's, and nothing downstream can tell from
        the weights alone, so the warning is the only thing standing between a
        salvaged encoder and a results table that silently mixes the two.
        """
        import logging

        import torch

        from iqssl.data.dataset import IQDataset
        from iqssl.eval.loading import load_encoder

        run_dir, root = pretrained_run
        ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert ckpt["step"] == ckpt["total_steps"], "fixture run should be complete"

        # Rewrite it as though the run had been killed at 40%.
        ckpt["step"] = int(0.4 * ckpt["total_steps"])
        torch.save(ckpt, run_dir / "checkpoint.pt")
        try:
            ds = IQDataset(root, "train")
            with caplog.at_level(logging.WARNING):
                load_encoder(run_dir, ds.crop_len, ds.num_primary_classes)
            assert "PARTIAL" in caplog.text
            assert "not comparable" in caplog.text
        finally:
            ckpt["step"] = ckpt["total_steps"]
            torch.save(ckpt, run_dir / "checkpoint.pt")

    def test_hash_mismatch_is_refused(self, pretrained_run, tmp_path):
        run_dir, _ = pretrained_run
        other = tmp_path / "other_ds"
        build_dataset(
            GeneratorConfig(n_samples=512, shard_size=512, difficulty="smoke", seed=99),
            other,
            overwrite=True,
            progress=False,
        )
        with pytest.raises(DatasetMismatch, match="wrong thing"):
            check_dataset_hash(run_dir, other)

    def test_missing_full_config_is_actionable(self, tmp_path):
        from iqssl.eval.loading import load_run_config

        with pytest.raises(FileNotFoundError, match="full-config snapshot"):
            load_run_config(tmp_path)


class TestEvaluateCLI:
    """The summary printer, which no test covered until it crashed.

    `evaluate_run` was well tested; `main()` was not, and the gap was exactly
    the seam between them. The printer ran *after* eval.json was written, so the
    failure mode was a command that had already done its work correctly and then
    exited nonzero — the sort of thing a scripted sweep reports as a failure
    while the artifacts on disk are perfectly fine.
    """

    def test_summary_prints_when_finetune_is_skipped(self, pretrained_run, capsys):
        from iqssl.cli.evaluate import main

        run_dir, root = pretrained_run
        assert main(["--run", str(run_dir), "--data", str(root), "--skip-finetune"]) == 0

        out = capsys.readouterr().out
        assert "SKIPPED" in out, "a report without finetune must say so on its face"
        assert "--" in out
        assert "nuisance R^2" in out

    def test_summary_prints_with_the_full_protocol(self, pretrained_run, capsys):
        from iqssl.cli.evaluate import main

        run_dir, root = pretrained_run
        assert main(["--run", str(run_dir), "--data", str(root)]) == 0

        out = capsys.readouterr().out
        assert "SKIPPED" not in out
        assert "finetune" in out
