"""Stage-1 gate: DSP property tests.

These assert *physics*, not implementation details. Each one is written so that
it would fail if the corresponding standard bug were present — the factor-of-2
in complex noise variance, the RRC's removable singularities, the PA whose
strength depends on upstream normalization, an RRC leaking into the CPM path.

The end-to-end test here is ``test_zero_ber_clean_path``: bits in, bits out,
zero errors through the whole linear modulator chain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from iqssl.dsp import channel, cpm, filters, impair, modulate, power
from iqssl.dsp.convert import complex_to_ri, ri_to_complex
from tests import tolerances as tol

SPS = 8
FS = 1e6


def _tone(freq_norm: float, length: int, batch: int = 1) -> torch.Tensor:
    n = torch.arange(length, dtype=torch.float32)
    phase = 2 * math.pi * freq_norm * n
    return torch.polar(torch.ones_like(phase), phase).unsqueeze(0).expand(batch, -1).contiguous()


class TestConversions:
    def test_round_trip(self):
        x = torch.randn(4, 2, 32)
        assert torch.allclose(complex_to_ri(ri_to_complex(x)), x)

    def test_channel_order_is_i_then_q(self):
        x = torch.zeros(1, 2, 4)
        x[0, 0] = 1.0  # I
        assert torch.allclose(ri_to_complex(x).real, torch.ones(1, 4))
        assert torch.allclose(ri_to_complex(x).imag, torch.zeros(1, 4))

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="2 channels"):
            ri_to_complex(torch.randn(4, 3, 32))


class TestRRC:
    @pytest.mark.parametrize("rolloff", [0.2, 0.25, 0.35, 0.5, 1.0])
    def test_taps_are_finite(self, rolloff):
        h = filters.rrc_taps(SPS, rolloff, 11)
        assert torch.isfinite(h).all()

    def test_singularity_lands_on_a_sample_at_default_params(self):
        # beta=0.25, sps=8 => t = T/(4*beta) = T, exactly 8 samples out.
        # This is why the analytic limit is not optional.
        t_sing = 1.0 / (4 * 0.25)
        assert abs(t_sing * SPS - round(t_sing * SPS)) < 1e-12

    @pytest.mark.parametrize("rolloff", [0.2, 0.25, 0.35, 0.5])
    def test_singularity_values_match_numerical_limit(self, rolloff):
        # Compare the analytic limits against the generic formula evaluated a
        # hair away from the singularity. This validates the limit itself,
        # rather than re-asserting the same closed form twice.
        beta = rolloff
        h0_analytic, hs_analytic = filters.rrc_singularity_values(beta)

        def generic(t: float) -> float:
            num = math.sin(math.pi * t * (1 - beta)) + 4 * beta * t * math.cos(
                math.pi * t * (1 + beta)
            )
            den = math.pi * t * (1 - (4 * beta * t) ** 2)
            return num / den

        assert generic(1e-7) == pytest.approx(h0_analytic, abs=tol.RRC_SINGULARITY_LIMIT)
        t_s = 1.0 / (4 * beta)
        near = (generic(t_s * (1 + 1e-9)) + generic(t_s * (1 - 1e-9))) / 2
        assert near == pytest.approx(hs_analytic, abs=tol.RRC_SINGULARITY_LIMIT)

    def test_unit_energy(self):
        h = filters.rrc_taps(SPS, 0.25, 11)
        assert float(torch.linalg.vector_norm(h)) == pytest.approx(1.0, abs=tol.RRC_ENERGY)

    def test_symmetric(self):
        h = filters.rrc_taps(SPS, 0.25, 11)
        assert torch.allclose(h, h.flip(0), atol=1e-6)

    def test_nyquist_isi_criterion(self):
        # RRC convolved with itself is a raised cosine, which must vanish at
        # every nonzero symbol instant. This is the real test of the taps:
        # it fails for a wrong normalization, a wrong rolloff, or a botched
        # singularity, none of which a finiteness check would catch.
        span = 16
        h = filters.rrc_taps(SPS, 0.25, span + 1).double()
        rc = torch.nn.functional.conv1d(
            h.view(1, 1, -1), h.flip(0).view(1, 1, -1), padding=h.numel() - 1
        ).flatten()
        centre = rc.numel() // 2
        peak = rc[centre]
        assert peak > 0
        for k in range(1, span // 2):
            assert abs(float(rc[centre + k * SPS] / peak)) < 1e-3

    def test_even_span_rejected(self):
        with pytest.raises(ValueError, match="odd"):
            filters.rrc_taps(SPS, 0.25, 10)


class TestPowerAndNoise:
    @pytest.mark.parametrize("snr_db", [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0])
    def test_awgn_hits_the_requested_snr(self, snr_db):
        g = torch.Generator().manual_seed(0)
        clean = _tone(0.03, 4096, batch=200)
        noisy, _ = power.add_awgn(clean, snr_db, generator=g)
        measured = power.measure_snr_db(clean, noisy).mean().item()
        assert measured == pytest.approx(snr_db, abs=tol.SNR_MEASURE_DB)

    def test_noise_variance_splits_evenly_across_i_and_q(self):
        # The classic off-by-2: sigma^2 is the total power of a complex sample,
        # so each real component carries sigma^2 / 2. Applying sigma per
        # component instead puts every SNR label 3 dB off.
        g = torch.Generator().manual_seed(1)
        clean = torch.zeros(1, 200_000, dtype=torch.complex64)
        clean += 1.0  # unit power, so noise power == 1/snr_lin
        noisy, noise_power = power.add_awgn(clean, 0.0, generator=g)
        noise = noisy - clean
        expected_component_var = float(noise_power[0]) / 2
        assert float(noise.real.var()) == pytest.approx(
            expected_component_var, rel=tol.NOISE_VARIANCE_SPLIT
        )
        assert float(noise.imag.var()) == pytest.approx(
            expected_component_var, rel=tol.NOISE_VARIANCE_SPLIT
        )

    def test_measured_power_override_is_respected(self):
        # The generator measures power over the samples it will keep, because
        # PA compression and fading both move it by several dB.
        g = torch.Generator().manual_seed(2)
        x = _tone(0.01, 1024, batch=8) * 5.0
        _, np_default = power.add_awgn(x, 10.0, generator=g)
        _, np_override = power.add_awgn(x, 10.0, generator=g, measured_power=torch.ones(8))
        assert float(np_override[0]) == pytest.approx(0.1, rel=1e-5)
        assert float(np_default[0]) != pytest.approx(float(np_override[0]), rel=1e-3)

    def test_es_n0_offset(self):
        assert power.es_n0_db(0.0, 8, 0.25) == pytest.approx(10 * math.log10(8 / 1.25), abs=1e-9)

    def test_normalize_rms_gives_unit_power(self):
        x = _tone(0.01, 512, batch=4) * 7.3
        assert torch.allclose(power.signal_power(power.normalize_rms(x)), torch.ones(4), atol=1e-5)

    def test_normalization_no_snr_leak(self):
        # If absolute received power survives normalization, it is a shortcut
        # feature perfectly correlated with SNR and every robustness claim in
        # the thesis is contaminated.
        #
        # Asserted in both directions: the leak must exist before normalization
        # (otherwise the test proves nothing) and must be gone after. Post-
        # normalization the correlation itself is meaningless -- buffer power is
        # identically 1.0, so any residual is float32 noise around a constant --
        # so the second assertion is on the spread, not the correlation.
        g = torch.Generator().manual_seed(3)
        n = 512
        snrs = torch.empty(n).uniform_(-20, 20, generator=g)
        clean = _tone(0.02, 1024, batch=n)
        noisy, _ = power.add_awgn(clean, snrs, generator=g)

        raw_power = power.signal_power(noisy)
        leak = torch.corrcoef(torch.stack([raw_power.log10(), snrs]))[0, 1]
        assert abs(float(leak)) > 0.9, "no leak to remove; the test would be vacuous"

        normed_power = power.signal_power(power.normalize_rms(noisy))
        assert float(normed_power.std()) < tol.SNR_LEAK_CORR
        assert torch.allclose(normed_power, torch.ones(n), atol=1e-4)


class TestImpairments:
    @pytest.mark.parametrize(
        ("gain_db", "phase_deg"),
        [(0.5, 2.0), (1.0, 5.0), (0.2, 1.0), (2.0, 10.0)],
    )
    def test_iq_imbalance_image_rejection(self, gain_db, phase_deg):
        # Transmit a single complex exponential; imbalance creates an image at
        # the mirror frequency. Its relative level must match |mu/nu|^2.
        # The tone must land exactly on an FFT bin. At f=0.1 with n_fft=4096 it
        # would not, and the resulting spectral leakage buries the image for the
        # smaller imbalances.
        n_fft = 4096
        k_sig = 400
        f = k_sig / n_fft
        x = _tone(f, n_fft)
        y = impair.iq_imbalance(x, gain_db, phase_deg)
        spec = torch.fft.fft(y[0]).abs()
        k_img = n_fft - k_sig
        measured = 20 * math.log10(float(spec[k_sig] / spec[k_img]))
        expected = float(impair.image_rejection_ratio_db(gain_db, phase_deg))
        assert measured == pytest.approx(expected, abs=tol.IRR_DB)

    def test_iq_imbalance_identity_when_ideal(self):
        x = _tone(0.1, 256)
        assert torch.allclose(impair.iq_imbalance(x, 0.0, 0.0), x, atol=1e-6)

    def test_dc_offset_relative_level(self):
        x = _tone(0.05, 4096) * 3.0
        y = impair.dc_offset(x, -30.0, -40.0, dbc=True)
        dc = (y - x)[0, 0]
        rms = float(power.rms(x)[0])
        assert 20 * math.log10(abs(float(dc.real)) / rms) == pytest.approx(
            -30.0, abs=tol.DC_OFFSET_DBC
        )

    def test_saleh_vanishes_at_large_backoff(self):
        x = _tone(0.05, 512) * 2.0
        y = impair.saleh(x, ibo_db=60.0)
        # alpha_a=2 means small-signal gain 2; compare shape after removing it.
        ratio = y / x
        assert torch.allclose(ratio, ratio.mean() * torch.ones_like(ratio), atol=tol.SALEH_IDENTITY)

    def test_saleh_compresses_at_low_backoff(self):
        x = _tone(0.05, 512)
        hard = impair.saleh(x, ibo_db=0.0)
        soft = impair.saleh(x, ibo_db=30.0)
        # AM/PM shifts phase more when driven harder.
        assert float(torch.angle(hard[0, 0] / x[0, 0])) > float(torch.angle(soft[0, 0] / x[0, 0]))

    def test_pa_ibo_scale_invariance(self):
        # The test that keeps PA strength from depending on upstream gain:
        # doubling the input at fixed IBO must double the output exactly.
        x = torch.randn(2, 1024, dtype=torch.complex64)
        y1 = impair.saleh(x, ibo_db=6.0)
        y2 = impair.saleh(x * 2.0, ibo_db=6.0)
        assert torch.allclose(y2, y1 * 2.0, atol=tol.SALEH_SCALE_INVARIANCE, rtol=1e-4)

    def test_phase_noise_variance_growth(self):
        # Wiener walk: Var(theta[n] - theta[0]) = 2*pi*linewidth*n/fs.
        g = torch.Generator().manual_seed(5)
        lw, ell, batch = 1000.0, 512, 2000
        x = torch.ones(batch, ell, dtype=torch.complex64)
        y = impair.phase_noise(x, lw, FS, random_initial_phase=False, generator=g)
        # Unwrap: by n=511 the walk has std ~1.8 rad, so a raw angle() wraps
        # past +/-pi often enough to bias the variance downward by ~7%.
        theta = torch.from_numpy(np.unwrap(torch.angle(y).numpy(), axis=-1))
        for n in (64, 256, 511):
            expected = 2 * math.pi * lw * n / FS
            assert float(theta[:, n].var()) == pytest.approx(expected, rel=tol.PHASE_NOISE_VAR_REL)

    def test_phase_noise_initial_phase_is_per_buffer_not_fixed(self):
        # A constant starting phase would be a trivial emitter-identity leak.
        g = torch.Generator().manual_seed(6)
        x = torch.ones(64, 128, dtype=torch.complex64)
        y = impair.phase_noise(x, 100.0, FS, generator=g)
        assert float(torch.angle(y[:, 0]).std()) > 1.0

    def test_emitter_chain_runs_and_stays_finite(self):
        g = torch.Generator().manual_seed(7)
        x = torch.randn(4, 512, dtype=torch.complex64)
        y = impair.apply_emitter_chain(
            x,
            iq_gain_db=0.4,
            iq_phase_deg=3.0,
            dc_i_dbc=-35.0,
            dc_q_dbc=-38.0,
            pa_ibo_db=7.0,
            pn_linewidth_hz=200.0,
            sample_rate_hz=FS,
            generator=g,
        )
        assert y.shape == x.shape
        assert torch.isfinite(y.real).all() and torch.isfinite(y.imag).all()


class TestInterpolationAndDelay:
    def test_zero_delay_is_identity(self):
        x = _tone(0.03, 512, batch=2)
        y = channel.apply_fractional_delay(x, 0.0)
        assert torch.allclose(y[:, 32:-32], x[:, 32:-32], atol=1e-4)

    @pytest.mark.parametrize("d", [0.25, 0.5, 0.75, 1.5])
    def test_fractional_delay_round_trip(self, d):
        x = _tone(0.03, 1024, batch=1)
        y = channel.apply_fractional_delay(channel.apply_fractional_delay(x, d), -d)
        interior = slice(64, -64)
        assert torch.allclose(y[:, interior], x[:, interior], atol=tol.FRAC_DELAY_ROUNDTRIP)

    @pytest.mark.parametrize("d", [0.3, 0.5, 1.25, 2.0])
    def test_fractional_delay_peak_location(self, d):
        # Cross-correlate against the original and parabolically interpolate the
        # peak: the measured shift must match the requested one.
        g = torch.Generator().manual_seed(8)
        n = 2048
        base = torch.randn(1, n, dtype=torch.complex64, generator=g)
        base = filters.filter_complex(base, filters.rrc_taps(8, 0.25, 11), mode="same")
        y = channel.apply_fractional_delay(base, d)

        a = base[0, 64:-64]
        b = y[0, 64:-64]
        corr = torch.fft.ifft(torch.fft.fft(b) * torch.fft.fft(a).conj()).abs()
        k = int(corr.argmax())
        y0, y1, y2 = (float(corr[(k + o) % corr.numel()]) for o in (-1, 0, 1))
        frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        assert (k + frac) == pytest.approx(d, abs=tol.FRAC_DELAY_PEAK_SAMPLES)

    def test_sample_rate_offset_shifts_progressively(self):
        # Clock drift accumulates: the offset at the end of the buffer is much
        # larger than at the start. A fixed delay would not do that.
        x = _tone(0.02, 2048)
        y = channel.apply_sample_rate_offset(x, 500.0)
        early = (y[0, 10] - x[0, 10]).abs()
        late = (y[0, 2000] - x[0, 2000]).abs()
        assert float(late) > float(early)

    def test_zero_ppm_is_identity(self):
        x = _tone(0.02, 512)
        assert torch.allclose(channel.apply_sample_rate_offset(x, 0.0), x, atol=1e-4)

    def test_interpolation_preserves_a_bandlimited_tone(self):
        x = _tone(0.05, 1024)
        y = channel.apply_fractional_delay(x, 0.5)
        assert float(y[0, 100:-100].abs().mean()) == pytest.approx(1.0, abs=1e-2)


class TestChannel:
    def test_cfo_shifts_the_spectrum(self):
        n_fft = 2048
        x = _tone(0.1, n_fft)
        y = channel.apply_cfo(x, 0.05)
        assert int(torch.fft.fft(y[0]).abs().argmax()) == pytest.approx(
            round(0.15 * n_fft), abs=tol.RESAMPLE_TONE_BINS
        )

    def test_cfo_round_trip(self):
        x = _tone(0.03, 512, batch=3)
        y = channel.apply_cfo(channel.apply_cfo(x, 0.02), -0.02)
        assert torch.allclose(y, x, atol=1e-5)

    def test_pdp_sums_to_one(self):
        # Unnormalized profiles would let fading covertly shift SNR.
        for n in (1, 3, 5, 8):
            assert float(channel.exponential_pdp(n).sum()) == pytest.approx(1.0, abs=1e-6)

    def test_rayleigh_tap_power_matches_profile(self):
        g = torch.Generator().manual_seed(9)
        spec = channel.TDLSpec(n_taps=4, k_factor_db=-100.0)
        gains, _ = channel.sample_taps(spec, 20000, SPS, generator=g)
        p = channel.exponential_pdp(4)
        measured = (gains.abs() ** 2).mean(0)
        assert torch.allclose(measured, p, rtol=tol.TAP_POWER_REL)

    def test_rayleigh_magnitude_distribution(self):
        from scipy import stats

        g = torch.Generator().manual_seed(10)
        spec = channel.TDLSpec(n_taps=1, k_factor_db=-100.0)
        gains, _ = channel.sample_taps(spec, 20000, SPS, generator=g)
        mag = gains[:, 0].abs().numpy()
        # E|h|^2 = 1 for a single tap => Rayleigh scale = 1/sqrt(2).
        _, p_value = stats.kstest(mag, "rayleigh", args=(0, math.sqrt(0.5)))
        assert p_value > tol.RAYLEIGH_KS_P

    @staticmethod
    def _k_from_moments(h: torch.Tensor) -> float:
        """Moment-based Rician K estimator.

        The specular component carries a *random* phase (it must: a fixed one
        would make the channel deterministic), so it contributes to the sample
        variance and mean-based estimators read K as zero. The fourth-moment
        ratio ``r = E|h|^4 / (E|h|^2)^2 = (K^2+4K+2)/(K+1)^2`` is phase-blind;
        inverting it gives ``K = ((2-r) + sqrt(2-r)) / (r-1)``. Sanity: r=2 for
        Rayleigh (K=0), r->1 as K->inf.
        """
        p = (h.abs() ** 2).double()
        r = float((p**2).mean() / p.mean() ** 2)
        r = min(r, 2.0 - 1e-12)
        return ((2 - r) + math.sqrt(2 - r)) / (r - 1)

    def test_moment_estimator_recovers_rayleigh(self):
        # Calibrates the estimator itself before trusting it on Rician taps.
        #
        # The bound is 0.15 rather than ~0 because the estimator is intrinsically
        # imprecise near K=0: there r -> 2, and K ~ sqrt(2-r), so a sampling
        # error of delta in the fourth-moment ratio shows up as sqrt(delta) in K.
        # At 200k samples se(r) ~ 0.006, which floors the estimate around 0.08.
        # That floor is a property of the statistic, not of the channel model --
        # the low-variance tap-power check above is what pins the Rayleigh case.
        g = torch.Generator().manual_seed(30)
        spec = channel.TDLSpec(n_taps=1, k_factor_db=-100.0)
        gains, _ = channel.sample_taps(spec, 200_000, SPS, generator=g)
        assert self._k_from_moments(gains[:, 0]) < 0.15

    @pytest.mark.parametrize("k_db", [0.0, 6.0, 10.0])
    def test_rician_k_factor(self, k_db):
        g = torch.Generator().manual_seed(11)
        spec = channel.TDLSpec(n_taps=1, k_factor_db=k_db)
        gains, _ = channel.sample_taps(spec, 200_000, SPS, generator=g)
        # The fourth-moment estimator is high-variance, so the tolerance here is
        # looser than the second-moment tap-power check above.
        assert self._k_from_moments(gains[:, 0]) == pytest.approx(
            10 ** (k_db / 10), rel=tol.RICIAN_K_MOMENT_REL
        )

    def test_rician_specular_component_is_los_tap_only(self):
        # Spreading K across every tap is the standard bug; the specular energy
        # must appear in tap 0 and nowhere else.
        g = torch.Generator().manual_seed(12)
        spec = channel.TDLSpec(n_taps=3, k_factor_db=12.0)
        gains, _ = channel.sample_taps(spec, 20000, SPS, generator=g)
        p = channel.exponential_pdp(3)
        for l in (1, 2):
            var = float(gains[:, l].real.var() + gains[:, l].imag.var())
            assert var == pytest.approx(float(p[l]), rel=0.05)

    def test_cir_is_a_delta_for_a_single_undelayed_unit_tap(self):
        gains = torch.ones(1, 1, dtype=torch.complex64)
        delays = torch.zeros(1, 1)
        cir = channel.build_cir(gains, delays)
        peak = int(cir[0].abs().argmax())
        assert float(cir[0].abs()[peak]) == pytest.approx(1.0, abs=1e-4)
        assert float(cir[0].abs().sum() - cir[0].abs()[peak]) < 1e-3

    def test_apply_tdl_shapes_and_finiteness(self):
        g = torch.Generator().manual_seed(13)
        x = torch.randn(4, 512, dtype=torch.complex64)
        y, gains, delays = channel.apply_tdl(x, channel.TDLSpec(n_taps=3), SPS, generator=g)
        assert y.shape[0] == 4
        assert y.shape[1] >= x.shape[1]
        assert torch.isfinite(y.abs()).all()
        assert gains.shape == (4, 3)
        assert delays.shape == (4, 3)

    def test_jakes_envelope_has_unit_mean_power(self):
        g = torch.Generator().manual_seed(14)
        env = channel.jakes_envelope(64, 2, 512, 1e-4, generator=g)
        assert float((env.abs() ** 2).mean()) == pytest.approx(1.0, rel=0.1)


class TestLinearModulations:
    @pytest.mark.parametrize("name", modulate.LINEAR_MODULATIONS)
    def test_constellation_has_unit_average_power(self, name):
        c = modulate.get_modulator(name).constellation()
        assert float((c.abs() ** 2).mean()) == pytest.approx(1.0, abs=tol.CONSTELLATION_UNIT_POWER)

    @pytest.mark.parametrize("name", modulate.LINEAR_MODULATIONS)
    def test_constellation_size_matches_bits(self, name):
        mod = modulate.get_modulator(name)
        assert mod.constellation().numel() == 2**mod.spec.bits_per_symbol

    @pytest.mark.parametrize("name", ["qpsk", "qam16", "qam64", "pam4", "psk8"])
    def test_gray_coding_neighbours_differ_by_one_bit(self, name):
        # Gray coding is what makes a symbol error cost ~1 bit at moderate SNR.
        mod = modulate.get_modulator(name)
        c = mod.constellation()
        d = (c.unsqueeze(0) - c.unsqueeze(1)).abs()
        d.fill_diagonal_(float("inf"))
        min_d = d.min()
        for i in range(c.numel()):
            for j in range(c.numel()):
                if i != j and d[i, j] <= min_d * 1.01:
                    assert bin(i ^ j).count("1") == 1, f"{name}: {i}<->{j} not Gray-adjacent"

    @pytest.mark.parametrize("name", modulate.LINEAR_MODULATIONS)
    def test_demap_inverts_map_without_noise(self, name):
        mod = modulate.get_modulator(name)
        syms = torch.arange(mod.order).unsqueeze(0)
        assert torch.equal(mod.demap(mod.map_symbols(syms)), syms)

    @pytest.mark.parametrize("name", modulate.LINEAR_MODULATIONS)
    def test_zero_ber_clean_path(self, name):
        # The single best end-to-end test of the linear chain: bits -> symbols
        # -> RRC -> matched filter -> symbol instants -> bits, with no errors.
        # Catches wrong group delay, wrong normalization, a broken constellation
        # or a mis-specified filter span, all at once.
        g = torch.Generator().manual_seed(20)
        mod = modulate.get_modulator(name)
        n_sym = 400
        syms = modulate.random_symbols(n_sym, mod.order, batch=4, generator=g)
        tx = mod.modulate(syms, sps=SPS, mode="full")
        rx = modulate.matched_filter_downsample(tx, n_sym, sps=SPS)
        decided = mod.demap(rx)
        bits_tx = modulate.symbols_to_bits(syms, mod.spec.bits_per_symbol)
        bits_rx = modulate.symbols_to_bits(decided, mod.spec.bits_per_symbol)
        ber = float((bits_tx != bits_rx).float().mean())
        assert ber == tol.CLEAN_PATH_BER

    def test_bits_symbols_round_trip(self):
        g = torch.Generator().manual_seed(21)
        syms = modulate.random_symbols(64, 16, batch=3, generator=g)
        bits = modulate.symbols_to_bits(syms, 4)
        assert torch.equal(modulate.bits_to_symbols(bits, 4), syms)

    def test_modulate_full_mode_keeps_transients(self):
        mod = modulate.get_modulator("qpsk")
        syms = torch.zeros(1, 100, dtype=torch.long)
        full = mod.modulate(syms, sps=SPS, mode="full")
        same = mod.modulate(syms, sps=SPS, mode="same")
        assert full.shape[-1] > same.shape[-1]


class TestCPM:
    @pytest.mark.parametrize("name", cpm.CPM_MODULATIONS)
    def test_constant_modulus(self, name):
        # If an RRC ever leaks into the CPM path, this fails immediately.
        # That is the whole reason the test exists.
        g = torch.Generator().manual_seed(22)
        mod = modulate.get_modulator(name)
        syms = modulate.random_symbols(200, mod.order, batch=4, generator=g)
        x = mod.modulate(syms, sps=SPS)
        assert torch.allclose(x.abs(), torch.ones_like(x.abs()), atol=tol.CPM_CONSTANT_MODULUS)

    @pytest.mark.parametrize("name", cpm.CPM_MODULATIONS)
    def test_phase_is_continuous(self, name):
        g = torch.Generator().manual_seed(23)
        mod = modulate.get_modulator(name)
        syms = modulate.random_symbols(200, mod.order, batch=2, generator=g)
        phi = mod.phase(syms, sps=SPS)
        # Max phase step is bounded by the per-symbol advance spread over sps.
        assert float(phi.diff(dim=-1).abs().max()) < math.pi / 2

    def test_msk_equals_cpfsk_at_h_half(self):
        # MSK is *defined* as CPFSK with h=0.5 and a REC pulse. Keeping it as a
        # separate registry entry risks the two configs drifting apart; this
        # pins them together.
        g = torch.Generator().manual_seed(24)
        syms = modulate.random_symbols(128, 2, batch=4, generator=g)
        msk = modulate.get_modulator("msk").modulate(syms, sps=SPS)
        cpfsk = modulate.get_modulator("cpfsk").modulate(syms, sps=SPS)
        assert torch.allclose(msk, cpfsk, atol=tol.MSK_VS_CPFSK)

    def test_phase_advance_per_symbol_matches_h(self):
        # Standard CPM normalization q(inf)=1/2 gives pi*h per symbol.
        # With area 1 instead, every modulation index would be doubled.
        syms = torch.ones(1, 8, dtype=torch.long)  # all +1 levels
        mod = modulate.get_modulator("cpfsk")
        phi = mod.phase(syms, sps=SPS)
        advance = float(phi[0, 2 * SPS] - phi[0, SPS])
        assert advance == pytest.approx(math.pi * mod.config.h, abs=1e-4)

    @pytest.mark.parametrize("name", cpm.FULL_RESPONSE_CPM)
    def test_zero_ber_clean_path_full_response(self, name):
        # Full-response CPM (REC pulse within one symbol) is exactly detectable
        # by a frequency discriminator. Partial-response GFSK is deliberately
        # excluded: its pulse spans 3 symbols, so ISI is by construction.
        g = torch.Generator().manual_seed(25)
        mod = modulate.get_modulator(name)
        n_sym = 300
        syms = modulate.random_symbols(n_sym, 2, batch=4, generator=g)
        x = mod.modulate(syms, sps=SPS)
        decided = mod.demod_differential(x, n_sym, sps=SPS)
        assert float((decided != syms).float().mean()) == tol.CLEAN_PATH_BER

    def test_gfsk_is_smoother_than_cpfsk(self):
        # The Gaussian pulse is what distinguishes GFSK; it must actually
        # band-limit the instantaneous frequency.
        g = torch.Generator().manual_seed(26)
        syms = modulate.random_symbols(200, 2, batch=4, generator=g)
        d_gfsk = modulate.get_modulator("gfsk").phase(syms, sps=SPS).diff(dim=-1)
        d_cpfsk = modulate.get_modulator("cpfsk").phase(syms, sps=SPS).diff(dim=-1)
        assert float(d_gfsk.diff(dim=-1).abs().mean()) < float(d_cpfsk.diff(dim=-1).abs().mean())

    def test_cpm_ignores_rrc_arguments(self):
        # The generator calls every modulator through one interface. CPM must
        # accept rolloff/span/mode and do nothing with them.
        syms = torch.zeros(1, 32, dtype=torch.long)
        mod = modulate.get_modulator("msk")
        a = mod.modulate(syms, sps=SPS)
        b = mod.modulate(syms, sps=SPS, rolloff=0.9, span=21, mode="same")
        assert torch.allclose(a, b)


def test_registry_covers_every_modulation():
    from iqssl.registry import MODULATORS

    modulate.get_modulator("bpsk")  # triggers cpm import
    expected = set(modulate.LINEAR_MODULATIONS) | set(cpm.CPM_MODULATIONS)
    assert expected <= set(MODULATORS.keys())
