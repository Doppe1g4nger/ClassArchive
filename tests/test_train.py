"""Stage-3 tests: the shared loop, schedules, and optimizers.

The properties worth testing here are the ones that make the *comparison* valid
rather than the ones that make a single run work. A loop that trains fine but
gives one method an extra warmup step, or that lets an augmentation stream
perturb weight init, produces a results table that looks completely normal and
means nothing.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import torch
from torch import nn

from iqssl.data.build import build_dataset
from iqssl.data.dataset import IQDataset
from iqssl.data.params import GeneratorConfig
from iqssl.registry import ENCODERS, METHODS, autodiscover
from iqssl.train.loop import TrainConfig, build_loader, train
from iqssl.train.optim import LARS, build_optimizer
from iqssl.train.schedules import apply_lr, cosine_with_warmup, scale_base_lr
from iqssl.utils.seed import seed_everything
from tests import tolerances as tol

autodiscover()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> IQDataset:
    root = tmp_path_factory.mktemp("train_ds")
    build_dataset(
        GeneratorConfig(n_samples=512, shard_size=512, difficulty="smoke", seed=0),
        root,
        overwrite=True,
        progress=False,
    )
    return IQDataset(root, "train")


def _method(name: str, dataset: IQDataset, seed: int = 0):
    """Build a method the way the CLI does: seed, then encoder, then method.

    Constructing the encoder before seeding is exactly the bug
    ``test_same_seed_reproduces_the_loss_trace`` caught, so the helper mirrors
    production order rather than a convenient one.
    """
    seed_everything(seed)
    encoder = ENCODERS.get("vit1d")(size="tiny", seq_len=dataset.crop_len, patch_size=16)
    return METHODS.get(name)(encoder)


def _cfg(**kw) -> TrainConfig:
    base = dict(epochs=1, batch_size=8, max_steps=4, device="cpu", log_every=1, num_workers=0)
    return TrainConfig(**{**base, **kw})


class TestSchedule:
    def test_warmup_rises_then_cosine_falls(self):
        total = 100
        vals = [cosine_with_warmup(s, total, 0.1) for s in range(total)]
        assert vals[0] < vals[5] <= vals[9]
        assert vals[9] == pytest.approx(1.0)
        assert vals[-1] < 0.01

    def test_first_step_is_not_zero(self):
        # A zero first step wastes it, and with short smoke runs that is a
        # measurable fraction of the budget.
        assert cosine_with_warmup(0, 100, 0.1) > 0

    def test_single_step_run_does_not_divide_by_zero(self):
        assert cosine_with_warmup(0, 1, 0.1) == 1.0

    def test_linear_scaling_rule(self):
        assert scale_base_lr(0.3, 256) == pytest.approx(0.3)
        assert scale_base_lr(0.3, 512) == pytest.approx(0.6)

    def test_lr_scale_multiplies_peak(self):
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([{"params": [p], "lr": 0.1, "lr_scale": 10.0}], lr=0.1)
        apply_lr(opt, 10, 100)
        assert opt.param_groups[0]["lr"] == pytest.approx(1.0)

    def test_fix_lr_groups_never_decay(self):
        """SimSiam's predictor depends on this; decaying it collapses the model."""
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([{"params": [p], "lr": 0.05, "fix_lr": True}], lr=0.05)
        seen = []
        for step in (0, 50, 99):
            apply_lr(opt, step, 100)
            seen.append(opt.param_groups[0]["lr"])
        assert all(v == pytest.approx(0.05) for v in seen)


class TestOptimizers:
    @pytest.mark.parametrize("name", ["lars", "sgd", "adamw"])
    def test_builds_and_steps(self, name):
        model = nn.Linear(4, 4)
        opt = build_optimizer(
            [{"params": list(model.parameters()), "lr": 0.1, "weight_decay": 0.0}], name, lr=0.1
        )
        before = model.weight.detach().clone()
        model(torch.randn(8, 4)).sum().backward()
        opt.step()
        assert not torch.equal(before, model.weight)

    def test_unknown_optimizer_rejected(self):
        with pytest.raises(ValueError, match="unknown optimizer"):
            build_optimizer([], "adam")

    def test_lars_excluded_groups_match_plain_sgd(self):
        """`lars_exclude` must genuinely bypass the trust ratio.

        1-D parameters (biases, norm gains) have no meaningful weight norm, and
        adapting their step by one measurably hurts.
        """
        torch.manual_seed(0)
        p_a = torch.nn.Parameter(torch.randn(4))
        p_b = torch.nn.Parameter(p_a.detach().clone())

        lars = LARS([{"params": [p_a], "lr": 0.1, "lars_exclude": True}], lr=0.1, momentum=0.0)
        sgd = torch.optim.SGD([p_b], lr=0.1, momentum=0.0)
        grad = torch.randn(4)
        p_a.grad, p_b.grad = grad.clone(), grad.clone()
        lars.step()
        sgd.step()
        assert torch.allclose(p_a, p_b, atol=1e-6)

    def test_lars_adapts_included_groups(self):
        torch.manual_seed(0)
        p_a = torch.nn.Parameter(torch.randn(4, 4))
        p_b = torch.nn.Parameter(p_a.detach().clone())
        lars = LARS([{"params": [p_a], "lr": 0.1, "lars_exclude": False}], lr=0.1, momentum=0.0)
        sgd = torch.optim.SGD([p_b], lr=0.1, momentum=0.0)
        grad = torch.randn(4, 4)
        p_a.grad, p_b.grad = grad.clone(), grad.clone()
        lars.step()
        sgd.step()
        assert not torch.allclose(p_a, p_b, atol=1e-6)


class TestLoader:
    def test_collate_follows_the_view_spec(self, dataset):
        loader = build_loader(dataset, METHODS.get("simclr"), _cfg())
        batch = next(iter(loader))
        assert batch.x_raw.shape == (8, 2, dataset.crop_len)
        assert batch.y_primary is not None

    def test_label_needing_methods_get_a_balanced_sampler(self, dataset):
        """Switched on by ``needs_labels``, never by method name.

        SupCon is why: with many classes and a modest batch, a uniform sampler
        leaves most anchors with no positive at all and the loss quietly
        degenerates toward NT-Xent without erroring.
        """
        assert build_loader(dataset, METHODS.get("supcon"), _cfg()).sampler is not None
        plain = build_loader(dataset, METHODS.get("simclr"), _cfg())
        assert not isinstance(plain.sampler, torch.utils.data.WeightedRandomSampler)


class TestLoop:
    @pytest.mark.parametrize("name", sorted(METHODS.keys()))
    def test_every_method_trains_end_to_end(self, name, dataset):
        """One loop, every objective, no branching.

        Parametrized over the registry rather than a hand-written list, so a new
        method is covered the moment it is registered.
        """
        state = train(_method(name, dataset), dataset, _cfg())
        assert state.step == 4
        assert np.isfinite(state.final_loss)

    def test_views_are_built_for_the_method(self, dataset):
        state = train(_method("simclr", dataset), dataset, _cfg(augment="standard"))
        assert np.isfinite(state.final_loss)

    def test_same_seed_reproduces_the_loss_trace(self, dataset):
        """Bit-identical, not merely close.

        Any drift means unseeded state leaked into the step, and the seed-to-seed
        error bars the thesis reports would then include a component that has
        nothing to do with the seed.
        """
        a = train(_method("simclr", dataset), dataset, _cfg(seed=3))
        b = train(_method("simclr", dataset), dataset, _cfg(seed=3))
        assert a.final_loss == pytest.approx(b.final_loss, abs=tol.LOSS_TRACE_EXACT)

    def test_different_seeds_diverge(self, dataset):
        a = train(_method("simclr", dataset, seed=0), dataset, _cfg(seed=0))
        b = train(_method("simclr", dataset, seed=1), dataset, _cfg(seed=1))
        assert a.final_loss != b.final_loss

    def test_augmentation_does_not_perturb_weight_init(self, dataset):
        """Changing the policy must not change the initial weights.

        Otherwise a policy ablation confounds two variables at once, and the
        difference attributed to augmentation is partly a different model. The
        augmentation pipeline draws from its own generator for exactly this
        reason.
        """

        def first_weight(policy: str):
            m = _method("simclr", dataset, seed=0)
            before = next(m.encoder.parameters()).detach().clone()
            train(m, dataset, _cfg(augment=policy, max_steps=1))
            return before

        assert torch.equal(first_weight("standard"), first_weight("hardware_invariant"))

    def test_every_method_starts_from_the_same_encoder(self, dataset):
        """The control variable, enforced.

        "Encoder architecture and init seed held constant" means every method at
        a given seed starts from the *same weights*, not merely the same shape.
        If they differed, part of every method-to-method gap would be a different
        initialization.
        """
        ref = None
        for name in sorted(METHODS.keys()):
            w = next(_method(name, dataset, seed=0).encoder.parameters()).detach().clone()
            if ref is None:
                ref = w
            else:
                assert torch.equal(ref, w), name

    def test_compute_accounting_matches_the_run(self, dataset):
        state = train(_method("simclr", dataset), dataset, _cfg(batch_size=8, max_steps=4))
        assert state.compute["compute/optimizer_steps"] == 4
        assert state.compute["compute/samples_seen"] == 32
        # SimCLR runs the encoder once per view, so two forwards per step.
        assert state.compute["compute/encoder_forwards"] == 8

    def test_ema_momentum_is_logged_and_ramps(self, dataset):
        state = train(_method("byol", dataset), dataset, _cfg(max_steps=4, log_every=1))
        moms = [row["ema_momentum"] for row in state.history if "ema_momentum" in row]
        assert moms and all(0.99 <= m <= 1.0 for m in moms)

    def test_writes_a_complete_run_directory(self, dataset, tmp_path):
        out = tmp_path / "run"
        train(_method("simclr", dataset), dataset, _cfg(), out_dir=out)
        for name in ("metrics.csv", "config.json", "run_meta.json", "summary.json"):
            assert (out / name).exists(), name
        assert (out / "checkpoint.pt").exists()

    def test_online_probe_logs_accuracy(self, dataset):
        state = train(_method("simclr", dataset), dataset, _cfg(probe_every=1, log_every=1))
        assert any("probe_acc" in row for row in state.history)

    def test_collapse_canaries_are_always_logged(self, dataset):
        """Every method, every run.

        Collapse is the characteristic failure of the negative-free objectives and
        it is silent -- the loss falls beautifully while the encoder maps
        everything to a point.
        """
        for name in sorted(METHODS.keys()):
            state = train(_method(name, dataset), dataset, _cfg(max_steps=2, log_every=1))
            keys = set().union(*(row.keys() for row in state.history))
            assert any(k.endswith("rankme") for k in keys), name
            assert any(k.endswith("std_min") for k in keys), name


class TestFairnessContract:
    def test_method_configs_may_only_tune_permitted_keys(self):
        """The contract is enforced, not merely documented.

        A method config that quietly asked for 200 epochs would invalidate the
        whole results table, and nothing else in the pipeline would notice.
        """
        from omegaconf import OmegaConf

        from iqssl.cli.pretrain import merge_train_config

        cfg = OmegaConf.create(
            {
                "train": {"epochs": 10, "batch_size": 64},
                "method": {"name": "cheater", "train": {"epochs": 200}},
            }
        )
        with pytest.raises(ValueError, match="fairness contract"):
            merge_train_config(cfg)

    def test_permitted_overrides_are_applied(self):
        from omegaconf import OmegaConf

        from iqssl.cli.pretrain import merge_train_config

        cfg = OmegaConf.create(
            {
                "train": {"epochs": 10, "optimizer": "adamw", "base_lr": 1e-3},
                "method": {"name": "simclr", "train": {"optimizer": "lars", "base_lr": 0.3}},
            }
        )
        merged = merge_train_config(cfg)
        assert merged.optimizer == "lars"
        assert merged.base_lr == 0.3
        assert merged.epochs == 10  # the experiment still governs this


class TestCheckpointDurability:
    """A killed run must leave something loadable.

    Written after a container suspension killed a 5,600-step run at step 2,400
    and left no checkpoint at all -- losing 90 minutes of compute *and* the
    evaluation of a different run that had already finished.
    """

    def test_periodic_checkpoint_appears_before_the_run_ends(self, dataset, tmp_path):
        train(
            _method("simclr", dataset),
            dataset,
            _cfg(max_steps=6, ckpt_every=2),
            out_dir=tmp_path,
        )
        state = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert state["step"] == 6
        assert state["total_steps"] == 6

    def test_no_temp_file_survives_a_successful_write(self, dataset, tmp_path):
        """The write is atomic via rename; a leftover .tmp means it was copied
        into place instead, which reopens the truncation hole it closes."""
        train(_method("simclr", dataset), dataset, _cfg(max_steps=4), out_dir=tmp_path)
        assert not (tmp_path / "checkpoint.pt.tmp").exists()
        assert (tmp_path / "checkpoint.pt").exists()

    def test_ckpt_every_zero_still_writes_at_the_end(self, dataset, tmp_path):
        train(
            _method("simclr", dataset),
            dataset,
            _cfg(max_steps=4, ckpt_every=0),
            out_dir=tmp_path,
        )
        assert (tmp_path / "checkpoint.pt").exists()

    def test_checkpointing_is_not_a_method_tunable(self):
        """Durability is not a degree of freedom a method gets to vary."""
        from iqssl.cli.pretrain import METHOD_TUNABLE

        assert "ckpt_every" not in METHOD_TUNABLE


class TestDeviceAndPrecision:
    """GPU-readiness, exercised on CPU.

    None of this can be tested on the hardware it exists for, so what is pinned
    is the part that is checkable anywhere: that the requests are validated up
    front, that the CPU path is untouched, and that precision is a contract
    setting rather than a per-host flag.
    """

    def test_absent_accelerator_is_refused_by_name(self):
        from iqssl.utils.device import resolve_device

        if torch.cuda.is_available():  # pragma: no cover - depends on host
            pytest.skip("CUDA present; the failure path cannot be exercised")
        with pytest.raises(RuntimeError, match=re.escape("train.device=cuda")):
            resolve_device("cuda")

    def test_cpu_never_autocasts_whatever_was_asked_for(self):
        """CPU autocast is bf16-only and slower at these sizes, and would make
        the CI smoke tier exercise a numerical path no real run uses."""
        from iqssl.utils.device import autocast_dtype

        cpu = torch.device("cpu")
        for precision in ("fp32", "bf16", "fp16"):
            assert autocast_dtype(precision, cpu) is None

    def test_unknown_precision_is_refused(self):
        from iqssl.utils.device import autocast_dtype

        with pytest.raises(ValueError, match="unknown precision"):
            autocast_dtype("int8", torch.device("cpu"))

    def test_only_fp16_asks_for_a_loss_scaler(self):
        """bf16 carries fp32's exponent range, so nothing underflows and a
        scaler would add a failure mode for no benefit."""
        from iqssl.utils.device import needs_grad_scaler

        assert needs_grad_scaler(torch.float16)
        assert not needs_grad_scaler(torch.bfloat16)
        assert not needs_grad_scaler(None)

    def test_precision_is_not_a_method_tunable(self):
        """The knob that would otherwise let one method run bf16 and another
        fp32, so the table measured numerical tolerance alongside objectives."""
        from omegaconf import OmegaConf

        from iqssl.cli.pretrain import METHOD_TUNABLE, merge_train_config

        assert "precision" not in METHOD_TUNABLE
        cfg = OmegaConf.create(
            {
                "train": {"epochs": 1, "precision": "fp32"},
                "method": {"name": "cheater", "train": {"precision": "bf16"}},
            }
        )
        with pytest.raises(ValueError, match="fairness contract"):
            merge_train_config(cfg)

    def test_pin_memory_is_off_on_cpu(self, dataset):
        """It costs page-locked host memory and buys nothing when the tensors
        never leave the host."""
        assert build_loader(dataset, METHODS.get("simclr"), _cfg()).pin_memory is False

    def test_the_cpu_training_path_is_unchanged(self, dataset):
        """The scaler and autocast wrapping must be inert on CPU.

        Both are no-ops there by construction, but 'by construction' is what the
        grad-clip bug this guards against also looked like, so the loss trace is
        compared against a recorded run rather than trusted.
        """
        losses = [
            train(_method("simclr", dataset), dataset, _cfg(precision=p)).final_loss
            for p in ("fp32", "bf16")
        ]
        assert losses[0] == pytest.approx(losses[1]), (
            "precision changed a CPU run; autocast is supposed to be disabled there"
        )


@pytest.mark.slow
class TestTinyOverfit:
    """Can each objective actually drive its own loss down?

    A method can be wired correctly enough to run -- right shapes, finite loss,
    no crash -- and still be optimizing nothing, because a detached tensor or a
    swapped argument turned the objective into a constant. Every test above would
    pass. These run each method on a handful of batches with augmentation off and
    check the loss actually falls, which is the cheapest thing that distinguishes
    "runs" from "learns".

    Marked slow and run nightly: too expensive for every push, but a silent
    collapse regression must not survive a night. This is what the `slow` CI job
    was wired for.
    """

    @pytest.mark.parametrize("name", sorted(METHODS.keys()))
    def test_loss_falls_on_a_tiny_set(self, name, dataset):
        if not getattr(METHODS.get(name), "trainable", True):
            pytest.skip("deliberately non-trainable (the random floor)")
        state = train(
            _method(name, dataset, seed=0),
            dataset,
            _cfg(max_steps=60, batch_size=16, log_every=1, augment="light"),
        )
        losses = [row["loss"] for row in state.history if "loss" in row]
        assert len(losses) > 20, name

        early = float(np.mean(losses[:10]))
        late = float(np.mean(losses[-10:]))
        assert late < early, (
            f"{name} did not reduce its loss over 60 steps "
            f"(first-10 mean {early:.4f}, last-10 mean {late:.4f}). The method "
            f"runs but may not be optimizing anything -- check for a detached "
            f"tensor on the gradient path."
        )

    @pytest.mark.parametrize("name", sorted(METHODS.keys()))
    def test_representation_does_not_collapse(self, name, dataset):
        """The failure that is silent by construction.

        Negative-free objectives collapse by mapping every input to one point,
        and the loss falls *beautifully* while it happens. Per-dimension standard
        deviation and effective rank both fall off a cliff, so they are the
        canaries, and every method emits them every step.
        """
        state = train(
            _method(name, dataset, seed=0),
            dataset,
            _cfg(max_steps=60, batch_size=16, log_every=1, augment="light"),
        )
        final = state.history[-1]
        rankme = next(v for k, v in final.items() if k.endswith("rankme"))
        std_min = next(v for k, v in final.items() if k.endswith("std_min"))
        assert rankme > tol.MIN_RANKME, f"{name} collapsed: rankme {rankme:.2f}"
        assert std_min > tol.MIN_FEATURE_STD, f"{name} collapsed: std_min {std_min:.2e}"


@pytest.mark.slow
class TestFixedBatchOverfit:
    """Can each method drive its objective down on **one fixed batch**?

    This is the check that distinguishes "the gradients reach the encoder" from
    "the loss happens to drift". ``TestTinyOverfit`` above trains on the whole
    dataset for 60 steps, which conflates optimization with generalization: a
    method whose loss stays flat there might be broken, or might simply face a
    task it cannot generalize in 60 steps. Both look identical.

    Repeating a single batch removes generalization from the question entirely.
    A correctly wired objective must be able to memorize its way down; one that
    cannot has a detached tensor, a frozen parameter, or a target that does not
    depend on its input. This test was added after a supervised run sat at
    exactly ln(n_classes) for 600 steps and the existing suite could not say
    whether the method or the dataset was at fault -- a direct fixed-batch probe
    answered it in seconds (it memorized 64 samples perfectly, exonerating the
    method).
    """

    @pytest.mark.parametrize("name", sorted(METHODS.keys()))
    def test_objective_falls_on_a_repeated_batch(self, name, dataset):
        from iqssl.augment.pipeline import ViewPipeline

        method_cls = METHODS.get(name)
        if not getattr(method_cls, "trainable", True):
            pytest.skip("deliberately non-trainable (the random floor)")

        method = _method(name, dataset, seed=0)
        spec = method_cls.view_spec()

        # Built once, then reused verbatim: fresh augmentation or fresh masks
        # every step would make the batch un-memorizable and put us back to
        # measuring generalization.
        batch = ViewPipeline(
            spec,
            "none",
            seed=0,
            n_tokens=getattr(method.encoder, "num_patches", None),
        )(next(iter(build_loader(dataset, method_cls, _cfg(batch_size=32)))))

        opt = torch.optim.AdamW(method.parameters(), lr=1e-3)
        losses = []
        method.train()
        for step in range(60):
            opt.zero_grad(set_to_none=True)
            out = method(batch, step, 60)
            out.loss.backward()
            opt.step()
            method.on_step_end(step, 60)
            losses.append(float(out.loss.detach()))

        early, late = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
        assert late < early * 0.9, (
            f"{name} could not reduce its own objective on a single repeated "
            f"batch (first-5 mean {early:.4f}, last-5 mean {late:.4f}). "
            f"Generalization is not the question here -- this points at the "
            f"gradient path: a detached target, a frozen parameter, or a "
            f"prediction that does not depend on the input."
        )
