"""Stage-4 tests: view transforms, policies, and masking.

The important test in this file is
:func:`test_hardware_invariant_destroys_the_fingerprint`. Everything else checks
that ops are well-behaved; that one checks that the benchmark's central claim is
actually implemented, and it is written so it *can* fail.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from iqssl.augment import ops
from iqssl.augment.masking import keep_indices, make_mask
from iqssl.augment.pipeline import ViewPipeline
from iqssl.augment.policies import POLICIES, get_policy
from iqssl.data import baselines
from iqssl.data.generator import SyntheticIQGenerator
from iqssl.data.params import GeneratorConfig
from iqssl.dsp import modulate as mod
from iqssl.dsp.convert import complex_to_ri
from iqssl.dsp.impair import apply_emitter_chain
from iqssl.dsp.power import normalize_rms
from iqssl.registry import AUG_OPS, autodiscover
from iqssl.types import Batch, ViewSpec

autodiscover()


@pytest.fixture(scope="module")
def signals():
    """Real generated buffers, with their emitter labels.

    White noise would not do: the whole question is whether an augmentation
    preserves or destroys a *fingerprint*, and only a signal that carries one can
    answer it.
    """
    gen = SyntheticIQGenerator(GeneratorConfig(n_samples=1024, difficulty="smoke", seed=0))
    iq, meta = gen.generate(0, 512)
    return torch.from_numpy(iq), meta["emitter_id"].to_numpy()


def _g(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _separability(x: torch.Tensor, labels: np.ndarray, feature: str) -> float:
    """One-way ANOVA F for a classical feature against emitter id.

    F is between-group variance over within-group variance: high means the
    feature still separates emitters, ~1 means it carries no group information at
    all. Used rather than a classifier accuracy because it is cheap, has no
    hyperparameters, and has a known null value.
    """
    col = baselines.CLASSICAL_FEATURE_NAMES.index(feature)
    f = baselines.classical_features(x)[:, col].numpy().astype(np.float64)
    groups = [f[labels == e] for e in np.unique(labels) if (labels == e).sum() > 1]

    grand = f.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / max(1, len(groups) - 1)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / max(1, len(f) - len(groups))
    return float(between / max(within, 1e-12))


class TestOps:
    @pytest.mark.parametrize("name", sorted(AUG_OPS.keys()))
    def test_shape_dtype_and_finiteness_preserved(self, name, signals):
        x, _ = signals
        x = x[:16]
        y = AUG_OPS.get(name)(x, _g())  # type: ignore[operator]
        assert y.shape == x.shape
        assert y.dtype == torch.float32
        assert torch.isfinite(y).all()

    @pytest.mark.parametrize("name", sorted(AUG_OPS.keys()))
    def test_deterministic_given_a_generator(self, name, signals):
        # Two runs from the same generator state must agree exactly, or a
        # resumed run diverges from an uninterrupted one.
        x, _ = signals
        x = x[:8]
        fn = AUG_OPS.get(name)
        assert torch.equal(fn(x, _g(7)), fn(x, _g(7)))  # type: ignore[operator]

    @pytest.mark.parametrize("name", sorted(AUG_OPS.keys()))
    def test_actually_changes_the_signal(self, name, signals):
        # An op that silently no-ops would leave a policy weaker than it reads,
        # and nothing else in the suite would notice.
        x, _ = signals
        x = x[:8]
        assert not torch.allclose(AUG_OPS.get(name)(x, _g()), x, atol=1e-6)  # type: ignore[operator]

    def test_strengths_are_drawn_per_sample(self, signals):
        # Per-batch strengths would correlate the augmentation across a batch,
        # and for a contrastive loss the batch *is* the comparison set.
        x, _ = signals
        y = ops.phase_rotate(x[:32], _g())
        delta = (y - x[:32]).flatten(1).norm(dim=1)
        assert float(delta.std()) > 1e-3

    def test_ops_do_not_touch_the_global_rng(self, signals):
        x, _ = signals
        torch.manual_seed(0)
        before = torch.randn(4)
        torch.manual_seed(0)
        ops.multipath(x[:8], _g())
        ops.renoise(x[:8], _g())
        assert torch.equal(before, torch.randn(4))


class TestPolicies:
    @pytest.mark.parametrize("name", sorted(POLICIES))
    def test_policy_runs_and_preserves_shape(self, name, signals):
        x, _ = signals
        y = get_policy(name)(x[:16], _g())
        assert y.shape == x[:16].shape
        assert torch.isfinite(y).all()

    def test_every_op_is_reachable_from_some_policy(self):
        """No orphaned ops.

        An op nothing references is untested in practice and drifts out of sync
        with the primitives it wraps.
        """
        used = {spec.name for p in POLICIES.values() for spec in p.specs}
        assert set(AUG_OPS.keys()) == used

    def test_none_is_the_identity(self, signals):
        x, _ = signals
        assert torch.equal(get_policy("none")(x[:8], _g()), x[:8])

    def test_policies_renormalize_to_unit_power(self, signals):
        x, _ = signals
        for name in ("light", "standard", "heavy", "hardware_invariant"):
            y = get_policy(name)(x[:16], _g())
            p = (y**2).sum(1).mean(-1)
            assert torch.allclose(p, torch.ones_like(p), atol=1e-4), name

    def test_two_views_differ(self, signals):
        x, _ = signals
        pol = get_policy("standard")
        g = _g()
        assert not torch.allclose(pol(x[:8], g), pol(x[:8], g))

    def test_only_the_control_is_flagged_as_touching_the_fingerprint(self):
        flagged = {n for n, p in POLICIES.items() if p.touches_fingerprint}
        assert flagged == {"hardware_invariant"}


@pytest.fixture(scope="module")
def two_emitters():
    """The same QPSK waveform put through two deliberately contrasting devices.

    Measuring the fingerprint against a *generated dataset* does not work, and
    the reason is worth recording: modulation dominates every classical feature.
    OOK has a nonzero mean by construction, 64QAM and GFSK have wildly different
    envelope statistics, and all of that variance sits inside each emitter group.
    An emitter-vs-feature statistic computed across mixed modulations therefore
    reads ~1 (no separation) even when the fingerprint is strong.

    Holding the modulation, the symbols and the channel fixed and varying only
    the device isolates what the policies are actually supposed to preserve or
    destroy.
    """
    torch.manual_seed(0)
    n = 96
    modulator = mod.get_modulator("qpsk")
    symbols = torch.randint(0, 4, (2 * n, 200))
    clean = modulator.modulate(symbols, sps=8, rolloff=0.25, span=11, mode="full")
    clean = clean[:, 88 : 88 + 1024]

    a = dict(iq_gain_db=2.5, iq_phase_deg=15.0, dc_i_dbc=-14.0, dc_q_dbc=-14.0, pa_ibo_db=2.0)
    b = dict(iq_gain_db=-2.5, iq_phase_deg=-15.0, dc_i_dbc=-28.0, dc_q_dbc=-28.0, pa_ibo_db=10.0)
    sig = torch.cat(
        [
            apply_emitter_chain(clean[:n], pn_linewidth_hz=120.0, sample_rate_hz=1e6, **a),
            apply_emitter_chain(clean[n:], pn_linewidth_hz=15.0, sample_rate_hz=1e6, **b),
        ]
    )
    labels = np.r_[np.zeros(n), np.ones(n)]
    return complex_to_ri(normalize_rms(sig)), labels


def _gap_over_view_jitter(x: torch.Tensor, labels: np.ndarray, policy: str) -> float:
    """Best emitter separation relative to view-to-view jitter, over all features.

    This, and not a marginal separability score, is what a contrastive objective
    actually responds to. The loss is asked to make two views of one buffer agree,
    so a feature whose view-to-view jitter swamps the between-emitter gap is a
    feature the objective is actively being told to discard. Above 1 the emitter
    signal survives the augmentation; below 1 the augmentation is louder than the
    label.
    """
    pol = get_policy(policy)
    g = _g()
    f1 = baselines.classical_features(pol(x, g)).numpy().astype(np.float64)
    f2 = baselines.classical_features(pol(x, g)).numpy().astype(np.float64)

    mean = (f1 + f2) / 2
    gap = np.abs(mean[labels == 0].mean(0) - mean[labels == 1].mean(0))
    jitter = (f1 - f2).std(0) + 1e-12
    return float((gap / jitter).max())


class TestPositiveControl:
    """The claim the augmentation design rests on, measured rather than asserted.

    One caveat is recorded here rather than hidden, because it bounds what these
    tests can prove. Augmentation is a *forward* operation: ``pa_jitter`` adds a
    second compression on top of the emitter's own, it cannot un-bake the first.
    So `hardware_invariant` reduces emitter separability substantially but does
    not drive it to zero, and a test demanding collapse would be demanding
    something the mechanism cannot deliver.

    The stronger claim -- that the control visibly destroys emitter *probe
    accuracy* -- is about a learned representation after training, where the
    objective actively optimizes for view invariance rather than merely inheriting
    marginal statistics. Verifying that needs a training run under both policies
    and belongs to the evaluation stage, not here.
    """

    def test_phase_and_timing_augmentation_is_free(self, two_emitters):
        """`light` must leave the fingerprint essentially untouched.

        Carrier phase, fractional timing and clock drift carry no hardware
        information, so a policy built only from them should cost nothing. If this
        ever fails, an op that touches transmitter hardware has leaked into the
        channel-side group.
        """
        x, labels = two_emitters
        assert _gap_over_view_jitter(x, labels, "light") > 20.0

    def test_the_control_is_measurably_more_destructive_than_standard(self, two_emitters):
        """The hardware ops must add destruction beyond the channel ops.

        `hardware_invariant` is `standard` plus three ops that randomize exactly
        the impairments the emitter label is made of, so it must separate from
        `standard` by a clear margin. If it does not, either the generator is not
        putting the fingerprint in those impairments or the jitter is too weak to
        dominate them -- and every emitter result would then be measuring
        something other than what it claims.
        """
        x, labels = two_emitters
        standard = _gap_over_view_jitter(x, labels, "standard")
        control = _gap_over_view_jitter(x, labels, "hardware_invariant")
        assert control < standard * 0.8, (
            f"`hardware_invariant` scored {control:.2f} against `standard`'s "
            f"{standard:.2f}; the hardware ops are not dominating the emitter "
            f"prior in iqssl/data/params.py. Widen their strengths, or check that "
            f"the fingerprint is where the generator claims it is."
        )

    def test_aggressive_policies_push_the_emitter_signal_below_the_noise(self, two_emitters):
        """`heavy` and the control both drive the ratio under 1.

        Below 1 the augmentation varies a feature more than the emitter does, which
        is the regime where a contrastive objective destroys it. `light` is far
        above 1, so the ladder genuinely spans the transition rather than sitting
        entirely on one side of it.
        """
        x, labels = two_emitters
        for policy in ("heavy", "hardware_invariant"):
            assert _gap_over_view_jitter(x, labels, policy) < 1.0, policy


class TestMasking:
    @pytest.mark.parametrize("kind", ["random", "block", "causal"])
    def test_shape_and_dtype(self, kind):
        m = make_mask(4, 64, kind, generator=_g())
        assert m.shape == (4, 64)
        assert m.dtype == torch.bool

    @pytest.mark.parametrize("kind", ["random", "block", "causal"])
    def test_never_masks_everything_or_nothing(self, kind):
        # A fully-masked sample has no context to predict from; a fully-visible
        # one makes the objective trivial. Either silently weakens the batch.
        m = make_mask(32, 64, kind, ratio=0.75, generator=_g())
        assert (m.sum(1) > 0).all()
        assert (m.sum(1) < 64).all()

    def test_random_masks_an_exact_count(self):
        # MAE gathers a fixed-width tensor of kept tokens, so a Bernoulli mask
        # with a variable count would ragged the batch.
        m = make_mask(16, 64, "random", ratio=0.75, generator=_g())
        assert m.sum(1).unique().numel() == 1
        assert int(m.sum(1)[0]) == 48

    def test_block_masks_are_contiguous_runs(self):
        m = make_mask(8, 64, "block", ratio=0.5, block_size=8, generator=_g())
        # Far fewer transitions than a random scatter of the same density.
        transitions = (m[:, 1:] != m[:, :-1]).sum(1).float().mean()
        rand = make_mask(8, 64, "random", ratio=0.5, generator=_g())
        assert transitions < (rand[:, 1:] != rand[:, :-1]).sum(1).float().mean() / 2

    def test_causal_masks_a_suffix(self):
        m = make_mask(16, 64, "causal", ratio=0.5, generator=_g())
        for row in m:
            first = int(row.float().argmax())
            assert row[first:].all(), "causal mask must be an unbroken suffix"

    def test_invalid_ratio_rejected(self):
        with pytest.raises(ValueError, match="mask ratio"):
            make_mask(2, 8, "random", ratio=1.0)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown mask kind"):
            make_mask(2, 8, "nope")  # type: ignore[arg-type]

    def test_keep_indices_round_trip(self):
        mask = make_mask(4, 32, "random", ratio=0.75, generator=_g())
        keep, restore = keep_indices(mask)
        assert keep.shape == (4, 8)
        # Gathering the kept positions then restoring must recover input order.
        full = torch.arange(32).unsqueeze(0).expand(4, -1)
        shuffled = torch.gather(full, 1, mask.long().argsort(dim=1, stable=True))
        assert torch.equal(torch.gather(shuffled, 1, restore), full)

    def test_keep_indices_rejects_ragged_masks(self):
        ragged = torch.zeros(2, 8, dtype=torch.bool)
        ragged[0, :3] = True
        ragged[1, :5] = True
        with pytest.raises(ValueError, match="uniform keep count"):
            keep_indices(ragged)


class TestPipeline:
    def _batch(self, x):
        return Batch(x_raw=x, y_primary=torch.zeros(len(x), dtype=torch.long))

    def test_builds_the_requested_number_of_views(self, signals):
        x, _ = signals
        for n in (0, 1, 2, 4):
            b = ViewPipeline(ViewSpec(n_views=n), "standard")(self._batch(x[:8]))
            assert len(b.views) == n

    def test_attaches_masks_when_requested(self, signals):
        x, _ = signals
        spec = ViewSpec(n_views=1, needs_mask=True, mask_kind="random")
        b = ViewPipeline(spec, "standard", n_tokens=16)(self._batch(x[:8]))
        assert b.masks is not None and b.masks["mask"].shape == (8, 16)

    def test_masking_without_n_tokens_is_rejected_at_construction(self):
        # Better a loud failure when the pipeline is built than a shape mismatch
        # deep inside an encoder on the first step.
        spec = ViewSpec(n_views=1, needs_mask=True, mask_kind="random")
        with pytest.raises(ValueError, match="n_tokens"):
            ViewPipeline(spec, "standard")

    def test_no_masks_when_not_requested(self, signals):
        x, _ = signals
        assert ViewPipeline(ViewSpec(), "standard")(self._batch(x[:8])).masks is None

    def test_same_seed_reproduces_the_view_sequence(self, signals):
        x, _ = signals
        a = ViewPipeline(ViewSpec(), "standard", seed=3)(self._batch(x[:8])).views
        b = ViewPipeline(ViewSpec(), "standard", seed=3)(self._batch(x[:8])).views
        assert all(torch.equal(p, q) for p, q in zip(a, b, strict=True))

    def test_state_round_trips_for_resume(self, signals):
        x, _ = signals
        p = ViewPipeline(ViewSpec(), "standard", seed=1)
        p(self._batch(x[:8]))
        state = p.state_dict()
        expected = p(self._batch(x[:8])).views

        p.load_state_dict(state)
        assert all(
            torch.equal(a, b) for a, b in zip(p(self._batch(x[:8])).views, expected, strict=True)
        )
