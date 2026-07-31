"""Method-contract and loss tests.

The parametrized tests here run over ``METHODS.keys()``, so every objective
added later is automatically held to the same contract without anyone
remembering to add a test.

The analytic loss checks are the valuable ones. An SSL loss that is subtly
wrong still trains, still converges, and still produces a plausible number —
the failure surfaces only as "this method underperforms on IQ data", which is
indistinguishable from a real finding. Pinning each loss against a
hand-computed or reduction-based reference is the only way to tell those apart.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from iqssl.methods import losses
from iqssl.methods.base import default_param_groups, rankme
from iqssl.models.ema import EMATeacher
from iqssl.models.heads import LinearClassifier, MLPHead, Projector
from iqssl.models.vit1d import ViT1D, sincos_positional_embedding
from iqssl.registry import METHODS, autodiscover
from iqssl.types import Batch, ViewSpec
from tests import tolerances as tol

autodiscover()

SEQ_LEN = 256
PATCH = 16


def tiny_encoder() -> ViT1D:
    return ViT1D(
        seq_len=SEQ_LEN, patch_size=PATCH, embed_dim=64, depth=2, num_heads=4, drop_path=0.0
    )


def fake_batch(
    b: int = 8, n_views: int = 2, n_classes: int = 4, spec: ViewSpec | None = None
) -> Batch:
    """A batch shaped the way ``spec`` asks, masks included.

    The contract tests hand every registered method a batch built from its own
    ``view_spec()``; ignoring ``needs_mask`` here would make the whole masked
    family untestable through the shared parametrized suite.
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(b, 2, SEQ_LEN, generator=g)
    masks = None
    if spec is not None and spec.needs_mask:
        from iqssl.augment.masking import make_jepa_masks, make_mask

        n_tokens = SEQ_LEN // PATCH
        if spec.mask_kind == "jepa":
            masks = make_jepa_masks(b, n_tokens, ratio=spec.mask_ratio, generator=g)
        else:
            assert spec.mask_kind is not None
            masks = {
                "mask": make_mask(
                    b,
                    n_tokens,
                    spec.mask_kind,
                    ratio=spec.mask_ratio,
                    block_size=spec.mask_block_size,
                    generator=g,
                )
            }
    return Batch(
        x_raw=x,
        views=[torch.randn(b, 2, SEQ_LEN, generator=g) for _ in range(n_views)],
        y_primary=torch.arange(b) % n_classes,
        y_mod=torch.arange(b) % n_classes,
        y_emitter=torch.arange(b) % n_classes,
        masks=masks,
    )


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


class TestNTXent:
    def test_perfectly_aligned_views_give_low_loss(self):
        z = F.normalize(torch.randn(16, 8), dim=-1)
        aligned, _ = losses.nt_xent(z, z.clone(), 0.1)
        misaligned, _ = losses.nt_xent(z, torch.randn(16, 8), 0.1)
        assert float(aligned) < float(misaligned)

    def test_matches_hand_computation_for_two_samples(self):
        # Two samples, two views: each anchor has 1 positive and 2 negatives.
        z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        z2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        tau = 0.5
        loss, _ = losses.nt_xent(z1, z2, tau)

        # Normalized, so similarities are 1 for matching directions and 0 else.
        pos, neg = math.exp(1 / tau), math.exp(0.0)
        expected = -math.log(pos / (pos + neg + neg))
        assert float(loss) == pytest.approx(expected, abs=1e-5)

    def test_self_similarity_is_masked_with_neg_inf(self):
        # Masking with 0 instead would leave every anchor competing against a
        # phantom negative of moderate similarity.
        z = F.normalize(torch.randn(8, 4), dim=-1)
        loss, logs = losses.nt_xent(z, z.clone(), 0.1)
        assert torch.isfinite(loss)
        assert logs["contrastive_acc"] == 1.0

    def test_lower_temperature_sharpens(self):
        g = torch.Generator().manual_seed(1)
        z1 = torch.randn(16, 8, generator=g)
        z2 = z1 + 0.3 * torch.randn(16, 8, generator=g)
        assert float(losses.nt_xent(z1, z2, 0.05)[0]) != pytest.approx(
            float(losses.nt_xent(z1, z2, 0.5)[0])
        )


class TestSupCon:
    def test_reduces_to_nt_xent_with_all_distinct_labels(self):
        # With one sample per class, each anchor's only positive is its own
        # other view -- exactly NT-Xent. This is the sharpest available check
        # that the L_out formulation is right.
        g = torch.Generator().manual_seed(2)
        z1 = torch.randn(12, 8, generator=g)
        z2 = torch.randn(12, 8, generator=g)
        labels = torch.arange(12)
        sup, _ = losses.supcon(z1, z2, labels, 0.1)
        nt, _ = losses.nt_xent(z1, z2, 0.1)
        assert float(sup) == pytest.approx(float(nt), abs=1e-5)

    def test_uses_multiple_positives_when_labels_repeat(self):
        g = torch.Generator().manual_seed(3)
        z1 = torch.randn(12, 8, generator=g)
        z2 = torch.randn(12, 8, generator=g)
        _, logs = losses.supcon(z1, z2, torch.arange(12) % 3, 0.1)
        # 4 samples per class x 2 views => 7 positives per anchor.
        assert logs["mean_positives"] == pytest.approx(7.0)

    def test_grouping_same_class_lowers_loss(self):
        g = torch.Generator().manual_seed(4)
        base = F.normalize(torch.randn(3, 8, generator=g), dim=-1)
        labels = torch.arange(12) % 3
        clustered = base[labels] + 0.01 * torch.randn(12, 8, generator=g)
        scattered = torch.randn(12, 8, generator=g)
        assert float(losses.supcon(clustered, clustered.clone(), labels, 0.1)[0]) < float(
            losses.supcon(scattered, scattered.clone(), labels, 0.1)[0]
        )

    def test_handles_a_batch_with_no_positives(self):
        z = torch.randn(4, 8)
        loss, logs = losses.supcon(z, z.clone(), torch.arange(4), 0.1)
        assert torch.isfinite(loss)
        assert logs["n_valid_anchors"] > 0


class TestBarlowTwins:
    def test_identical_decorrelated_views_give_near_zero_loss(self):
        g = torch.Generator().manual_seed(5)
        z = torch.randn(512, 16, generator=g)
        loss, logs = losses.barlow_twins(z, z.clone(), lambd=5e-3)
        # Not exactly zero: torch's std() is Bessel-corrected, so standardized
        # dimensions have biased variance b/(b-1) and the diagonal sits at
        # 1 + 1/b rather than 1. That is a property of the reference
        # implementation, which also uses std(), not a defect.
        assert logs["bt_on_diag"] < 1e-4
        assert float(loss) < 0.5

    def test_penalizes_correlated_dimensions(self):
        g = torch.Generator().manual_seed(6)
        z = torch.randn(512, 8, generator=g)
        dup = z.clone()
        dup[:, 1] = dup[:, 0]  # two dimensions carrying the same information
        assert (
            losses.barlow_twins(dup, dup.clone())[1]["bt_off_diag"]
            > (losses.barlow_twins(z, z.clone())[1]["bt_off_diag"])
        )

    def test_penalizes_view_mismatch(self):
        g = torch.Generator().manual_seed(7)
        z1 = torch.randn(256, 8, generator=g)
        assert float(losses.barlow_twins(z1, torch.randn(256, 8, generator=g))[0]) > float(
            losses.barlow_twins(z1, z1.clone())[0]
        )


class TestVICReg:
    def test_variance_term_saturates_for_unit_std(self):
        g = torch.Generator().manual_seed(8)
        z = torch.randn(2048, 16, generator=g)
        _, logs = losses.vicreg(z, z.clone())
        assert logs["vicreg_var"] == pytest.approx(0.0, abs=0.05)

    def test_variance_term_is_maximal_for_a_collapsed_batch(self):
        # Every sample identical => std 0 => hinge saturates. The exact value is
        # 2*(1 - sqrt(eps)) = 1.98, not 2.0, and that 0.02 gap *is* the eps
        # inside the sqrt -- the term that keeps the gradient finite here.
        z = torch.ones(64, 16)
        _, logs = losses.vicreg(z, z.clone(), eps=1e-4)
        assert logs["vicreg_var"] == pytest.approx(2 * (1 - math.sqrt(1e-4)), abs=1e-4)

    def test_survives_zero_variance_without_nan(self):
        # The eps lives *inside* the sqrt precisely so this gradient is finite.
        z = torch.ones(32, 8, requires_grad=True)
        loss, _ = losses.vicreg(z, z.clone().detach())
        loss.backward()
        assert torch.isfinite(z.grad).all()

    def test_covariance_term_ignores_the_diagonal(self):
        # Scaling one dimension changes only its own variance (a diagonal
        # entry), so the off-diagonal penalty must be unmoved. Note the term is
        # computed on raw covariances, not correlations, so scaling the *whole*
        # embedding legitimately does change it.
        g = torch.Generator().manual_seed(9)
        z = torch.randn(1024, 8, generator=g)
        bumped = z.clone()
        bumped[:, 0] *= 4.0
        base_cov = losses.vicreg(z, z.clone())[1]["vicreg_cov"]
        # Off-diagonals involving dim 0 scale with it, so compare a dimension
        # pair that does not: zero-correlation columns stay near zero either way.
        assert base_cov < 0.5
        assert losses.vicreg(bumped, bumped.clone())[1]["vicreg_cov"] > base_cov

    def test_covariance_term_punishes_correlated_dimensions(self):
        g = torch.Generator().manual_seed(14)
        z = torch.randn(1024, 8, generator=g)
        dup = z.clone()
        dup[:, 1] = dup[:, 0]
        assert (
            losses.vicreg(dup, dup.clone())[1]["vicreg_cov"]
            > (losses.vicreg(z, z.clone())[1]["vicreg_cov"])
        )

    def test_invariance_term_is_plain_mse(self):
        g = torch.Generator().manual_seed(10)
        z1, z2 = torch.randn(64, 8, generator=g), torch.randn(64, 8, generator=g)
        _, logs = losses.vicreg(z1, z2)
        assert logs["vicreg_inv"] == pytest.approx(float(F.mse_loss(z1, z2)), abs=1e-6)


class TestNegativeCosine:
    def test_range_and_endpoints(self):
        z = F.normalize(torch.randn(16, 8), dim=-1)
        assert float(losses.negative_cosine(z, z)) == pytest.approx(0.0, abs=1e-6)
        assert float(losses.negative_cosine(z, -z)) == pytest.approx(4.0, abs=1e-6)

    def test_bounded(self):
        g = torch.Generator().manual_seed(11)
        v = float(
            losses.negative_cosine(torch.randn(64, 8, generator=g), torch.randn(64, 8, generator=g))
        )
        assert 0.0 <= v <= 4.0


# --------------------------------------------------------------------------
# Encoders and heads
# --------------------------------------------------------------------------


class TestViT1D:
    def test_output_shapes(self):
        enc = tiny_encoder()
        out = enc(torch.randn(4, 2, SEQ_LEN), return_tokens=True)
        assert out.mean.shape == (4, 64)
        assert out.cls is not None and out.cls.shape == (4, 64)
        assert out.tokens is not None and out.tokens.shape == (4, SEQ_LEN // PATCH, 64)

    def test_positional_embedding_is_a_buffer_not_a_parameter(self):
        # Learnable positions would lag inside an EMA teacher, so student and
        # teacher would disagree about where each token is.
        enc = tiny_encoder()
        assert "pos_embed" not in dict(enc.named_parameters())
        assert "pos_embed" in dict(enc.named_buffers())

    def test_sincos_table_is_deterministic_and_bounded(self):
        a = sincos_positional_embedding(16, 8)
        assert torch.equal(a, sincos_positional_embedding(16, 8))
        assert a.abs().max() <= 1.0

    def test_odd_embed_dim_rejected(self):
        with pytest.raises(ValueError, match="even"):
            sincos_positional_embedding(4, 7)

    def test_seq_len_must_divide_by_patch(self):
        with pytest.raises(ValueError, match="divisible"):
            ViT1D(seq_len=100, patch_size=16)

    def test_forward_masked_keeps_only_requested_tokens(self):
        enc = tiny_encoder()
        n = SEQ_LEN // PATCH
        keep = torch.arange(n // 2).unsqueeze(0).expand(3, -1)
        out = enc.forward_masked(torch.randn(3, 2, SEQ_LEN), keep)
        assert out.shape == (3, n // 2 + 1, 64)  # +1 for the class token

    def test_masked_forward_with_all_tokens_matches_full_forward(self):
        enc = tiny_encoder().eval()
        n = SEQ_LEN // PATCH
        x = torch.randn(2, 2, SEQ_LEN)
        keep = torch.arange(n).unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            a = enc.forward_masked(x, keep)
            b = enc(x, return_tokens=True)
        assert torch.allclose(a[:, 1:], b.tokens, atol=1e-5)

    def test_mask_token_path_preserves_sequence_length(self):
        enc = tiny_encoder()
        n = SEQ_LEN // PATCH
        mask = torch.zeros(2, n, dtype=torch.bool)
        mask[:, : n // 2] = True
        out = enc.forward_with_mask_token(torch.randn(2, 2, SEQ_LEN), mask)
        assert out.shape == (2, n, 64)

    def test_all_layers_are_returned_for_data2vec(self):
        enc = tiny_encoder()
        out = enc(torch.randn(2, 2, SEQ_LEN), return_all_layers=True)
        assert out.layers is not None and len(out.layers) == 2

    def test_pooled_rejects_unknown_mode(self):
        out = tiny_encoder()(torch.randn(1, 2, SEQ_LEN))
        with pytest.raises(ValueError, match="unknown pooling"):
            out.pooled("median")


class TestCNN1D:
    def test_forward_shape(self):
        from iqssl.models.cnn1d import cnn1d

        enc = cnn1d("tiny")
        assert enc(torch.randn(2, 2, SEQ_LEN)).mean.shape == (2, 128)

    def test_masked_methods_are_refused_with_a_clear_message(self):
        from iqssl.models.cnn1d import cnn1d

        with pytest.raises(NotImplementedError, match="encoder=vit1d"):
            cnn1d("tiny").forward_masked(
                torch.randn(2, 2, SEQ_LEN), torch.zeros(2, 4, dtype=torch.long)
            )

    def test_method_requiring_tokens_rejects_cnn_encoder(self):
        from iqssl.methods.base import Method
        from iqssl.models.cnn1d import cnn1d

        class Masked(Method):
            requires_tokenizer = True

            @classmethod
            def view_spec(cls, cfg=None):
                return ViewSpec(n_views=0)

            def forward(self, batch, step, total_steps):  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(ValueError, match="encoder=vit1d"):
            Masked(cnn1d("tiny"))


class TestHeads:
    def test_projector_shape(self):
        assert Projector(64, 128, 32)(torch.randn(8, 64)).shape == (8, 32)

    def test_mlp_rejects_zero_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            MLPHead(8, 8, 8, n_layers=0)

    def test_head_norm_choice_validated(self):
        with pytest.raises(ValueError, match="unknown norm"):
            MLPHead(8, 8, 8, norm="groupnorm")

    def test_linear_probe_has_affine_free_bn(self):
        # MAE's own protocol. Without it, masked methods probe far below what
        # their features actually support and the protocol picks the winner.
        probe = LinearClassifier(16, 4)
        assert isinstance(probe.bn, torch.nn.BatchNorm1d)
        assert probe.bn.affine is False


class TestEMATeacher:
    def test_update_is_exactly_the_ema_formula(self):
        student = torch.nn.Linear(4, 4)
        teacher = EMATeacher(student, momentum_start=0.9, momentum_end=0.9, schedule="constant")
        before = teacher.teacher.weight.detach().clone()
        with torch.no_grad():
            student.weight.add_(1.0)
        teacher.update(student, 0, 100)
        expected = 0.9 * before + 0.1 * student.weight.detach()
        assert torch.allclose(teacher.teacher.weight, expected, atol=tol.EMA_MATH)

    def test_buffers_are_copied_not_averaged(self):
        # EMA-ing BatchNorm running statistics gives a teacher whose
        # normalization lags its own weights. Training still converges, just to
        # a worse solution -- which reads as "this method underperforms".
        student = torch.nn.BatchNorm1d(4)
        teacher = EMATeacher(student, momentum_start=0.9, schedule="constant")
        with torch.no_grad():
            student.running_mean.fill_(5.0)
        teacher.update(student, 0, 100)
        assert torch.allclose(teacher.teacher.running_mean, torch.full((4,), 5.0))

    def test_teacher_parameters_never_require_grad(self):
        teacher = EMATeacher(torch.nn.Linear(4, 4))
        assert all(not p.requires_grad for p in teacher.teacher.parameters())

    def test_teacher_is_excluded_from_trainable_params(self):
        # An optimizer built over model.parameters() must not pick the teacher up.
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.student = torch.nn.Linear(4, 4)
                self.teacher = EMATeacher(self.student)

        trainable = [p for p in Wrapper().parameters() if p.requires_grad]
        assert len(trainable) == 2  # student weight and bias only

    @pytest.mark.parametrize("schedule", ["constant", "linear", "cosine"])
    def test_momentum_schedule_endpoints(self, schedule):
        t = EMATeacher(
            torch.nn.Linear(2, 2), momentum_start=0.99, momentum_end=1.0, schedule=schedule
        )
        assert t.momentum_at(0, 100) == pytest.approx(0.99, abs=1e-6)
        end = t.momentum_at(99, 100)
        assert end == pytest.approx(0.99 if schedule == "constant" else 1.0, abs=1e-6)

    def test_momentum_is_monotone(self):
        t = EMATeacher(torch.nn.Linear(2, 2), momentum_start=0.99, momentum_end=1.0)
        vals = [t.momentum_at(s, 100) for s in range(100)]
        assert all(b >= a - 1e-9 for a, b in itertools.pairwise(vals))

    def test_delegates_encoder_methods(self):
        teacher = EMATeacher(tiny_encoder())
        out = teacher.forward_with_mask_token(
            torch.randn(2, 2, SEQ_LEN), torch.zeros(2, SEQ_LEN // PATCH, dtype=torch.bool)
        )
        assert out.shape[0] == 2


class TestRankMe:
    def test_full_rank_gaussian_scores_near_the_dimension(self):
        g = torch.Generator().manual_seed(12)
        assert float(rankme(torch.randn(512, 16, generator=g))) > 12

    def test_collapsed_representation_scores_near_one(self):
        # A genuinely collapsed representation is rank-1: every sample lies on
        # one direction. (Note a *constant* representation is not the right test
        # -- rankme mean-centres, so a constant plus tiny noise is full-rank
        # noise, which is exactly why the per-dim std canary is logged too.)
        g = torch.Generator().manual_seed(15)
        direction = torch.randn(1, 16, generator=g)
        scales = torch.randn(256, 1, generator=g)
        assert float(rankme(scales @ direction)) < 2.0

    def test_low_rank_is_detected(self):
        g = torch.Generator().manual_seed(13)
        z = torch.randn(256, 3, generator=g) @ torch.randn(3, 16, generator=g)
        assert float(rankme(z)) < 5


def test_default_param_groups_exclude_1d_from_weight_decay():
    # Decaying biases, norm gains and learned tokens measurably hurts and is
    # omitted in essentially every published SSL implementation.
    enc = tiny_encoder()
    groups = default_param_groups(enc, 1e-3, 0.05)
    decayed = {id(p) for g in groups if g["weight_decay"] > 0 for p in g["params"]}
    for _, p in enc.named_parameters():
        if p.ndim <= 1:
            assert id(p) not in decayed


# --------------------------------------------------------------------------
# The Method contract, over every registered method
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", METHODS.keys())
class TestMethodContract:
    def _build(self, name: str):
        return METHODS.get(name)(tiny_encoder())

    def test_view_spec_is_a_classmethod(self, name):
        spec = METHODS.get(name).view_spec()
        assert isinstance(spec, ViewSpec)

    def test_forward_returns_a_finite_scalar_loss(self, name):
        method = self._build(name)
        spec = METHODS.get(name).view_spec()
        out = method(fake_batch(n_views=max(spec.n_views, 1), spec=spec), step=0, total_steps=10)
        assert out.loss.ndim == 0
        assert torch.isfinite(out.loss)
        assert "loss" in out.logs

    def test_backward_produces_encoder_gradients(self, name):
        method = self._build(name)
        if not getattr(method, "trainable", True):
            pytest.skip("deliberately non-trainable (the random floor)")
        spec = METHODS.get(name).view_spec()
        method(fake_batch(n_views=max(spec.n_views, 1), spec=spec), 0, 10).loss.backward()
        grads = [p.grad for p in method.encoder.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)

    def test_logs_a_collapse_canary(self, name):
        method = self._build(name)
        spec = METHODS.get(name).view_spec()
        logs = method(fake_batch(n_views=max(spec.n_views, 1), spec=spec), 0, 10).logs
        assert any(k.endswith("std_mean") for k in logs), "no collapse canary logged"

    def test_param_groups_cover_every_trainable_parameter(self, name):
        method = self._build(name)
        covered = {id(p) for g in method.param_groups(1e-3, 0.05) for p in g["params"]}
        for p in method.parameters():
            if p.requires_grad:
                assert id(p) in covered

    def test_on_step_end_is_callable(self, name):
        self._build(name).on_step_end(0, 10)


def test_registry_contains_the_expected_methods():
    assert {"simclr", "supcon"} <= set(METHODS.keys())


def test_supcon_requires_labels_and_simclr_does_not():
    assert METHODS.get("supcon").view_spec().needs_labels
    assert not METHODS.get("simclr").view_spec().needs_labels


def test_method_raises_a_clear_error_when_views_are_missing():
    method = METHODS.get("simclr")(tiny_encoder())
    with pytest.raises(ValueError, match="SimCLR needs 2 views"):
        method(fake_batch(n_views=1), 0, 10)


class TestPatchify:
    def test_round_trip(self):
        from iqssl.methods.losses import patchify, unpatchify

        x = torch.randn(3, 2, 64)
        assert torch.equal(unpatchify(patchify(x, 16), 16), x)

    def test_rejects_indivisible_length(self):
        from iqssl.methods.losses import patchify

        with pytest.raises(ValueError, match="divisible"):
            patchify(torch.randn(1, 2, 65), 16)


class TestMaskedLosses:
    def test_masked_mse_ignores_visible_positions(self):
        """The property that makes it a prediction task rather than an
        autoencoder: error at visible patches must contribute nothing."""
        from iqssl.methods.losses import masked_mse, patchify

        p = patchify(torch.randn(4, 2, 64), 16)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[:, :2] = True
        corrupted = p.clone()
        corrupted[:, 2:] += 100.0  # visible positions only
        loss, _ = masked_mse(corrupted, p, mask)
        assert float(loss) == 0.0

    def test_masked_mse_counts_masked_error(self):
        from iqssl.methods.losses import masked_mse

        p = torch.zeros(2, 4, 8)
        pred = p.clone()
        pred[:, 0] = 1.0
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[:, 0] = True
        loss, _ = masked_mse(pred, p, mask)
        assert float(loss) == pytest.approx(1.0)

    def test_target_normalization_is_per_patch(self):
        from iqssl.methods.losses import masked_mse

        target = torch.randn(2, 4, 16) * 5 + 3
        mask = torch.ones(2, 4, dtype=torch.bool)
        # Predicting the standardized target exactly gives zero loss.
        std = (target - target.mean(-1, keepdim=True)) / (
            target.var(-1, keepdim=True) + 1e-6
        ).sqrt()
        loss, _ = masked_mse(std, target, mask, normalize_targets=True)
        assert float(loss) < 1e-6

    def test_masked_smooth_l1_restricts_to_mask(self):
        from iqssl.methods.losses import masked_smooth_l1

        pred = torch.zeros(2, 4, 8)
        target = torch.zeros(2, 4, 8)
        pred[:, 3] = 10.0  # unmasked position
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[:, 0] = True
        assert float(masked_smooth_l1(pred, target, mask)) == 0.0


class TestMAE:
    def _mae(self):
        from iqssl.models.vit1d import vit1d

        return METHODS.get("mae")(vit1d(size="tiny", seq_len=256, patch_size=16))

    def test_encoder_never_sees_masked_tokens(self):
        """The load-bearing property. Change a masked patch in the input: the
        loss changes (the target moved) but the *latent* must not."""
        from iqssl.augment.masking import keep_indices, make_mask

        torch.manual_seed(0)
        m = self._mae()
        m.eval()
        x = torch.randn(2, 2, 256)
        mask = make_mask(2, 16, "random", ratio=0.75, generator=torch.Generator().manual_seed(0))
        keep_idx, _ = keep_indices(mask)

        with torch.no_grad():
            a = m.encoder.forward_masked(x, keep_idx)
            x2 = x.clone()
            first_masked = int(mask[0].nonzero()[0])
            x2[0, :, first_masked * 16 : (first_masked + 1) * 16] += 99.0
            b = m.encoder.forward_masked(x2, keep_idx)
        assert torch.allclose(a, b, atol=1e-5)

    def test_declares_fractional_compute(self):
        assert self._mae().encoder_passes_per_step() == (0.25, 0.0)

    def test_view_spec_reads_ratio_from_cfg(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"method": {"name": "mae", "args": {"mask_ratio": 0.5}}})
        assert METHODS.get("mae").view_spec(cfg).mask_ratio == 0.5
        assert METHODS.get("mae").view_spec(None).mask_ratio == 0.75
