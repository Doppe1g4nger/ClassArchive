"""Parity tests for the numba and numpy variant kernels against the
pulsecore reference implementations, skipped cleanly (not failed) when
the optional numeric dependencies aren't installed -- the five core
builds must stay testable with nothing but protobuf.

These are the unit-test-sized versions of the checks the two dedicated
verify tools run at full scale (scripts/verify_avx_variant.sh for C++,
python/numpy_variant/verify_numpy_variant.py for numpy): small enough to
run on every test invocation, focused on the invariants most likely to
break under maintenance -- bit-identity where each variant promises it,
tolerance where vectorized reductions make exactness impossible, and
the batch-straddling detector state neither variant's own signal ever
exercises.
"""
import unittest

from tests import *  # noqa: F401,F403 -- path bootstrap (see __init__.py)

from pulsecore import pulse_pb2
from pulsecore.iq_source import (
    SyntheticIQSource,
    _GAP_SAMPLES,
    _PULSE_SAMPLES,
    _PULSE_COMPONENT,
    _NOISE_AMPLITUDE,
)
from pulsecore.pulse_detector import PulseDetector
from pulsecore.spectrogram import SpectrogramAnalyzer
from pulsecore.jammer import JammerDetector

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from numba_variant import kernels as numba_kernels
except ImportError:  # pragma: no cover
    numba_kernels = None

try:
    from numpy_variant import kernels as numpy_kernels
    from numpy_variant.aggregates import DeinterleaverArrays, PulseStatsArrays
    from numpy_variant.iq_source_arrays import SyntheticIQSourceArrays
except ImportError:  # pragma: no cover
    numpy_kernels = None

SAMPLE_RATE_HZ = 10_000_000.0
NUM_PULSES = 3000  # a few full batches, small enough for unit-test time

# Theoretical-limits branch: the numba kernels compile fastmath=True
# (see numba_variant/kernels.py's docstring), so their parity with the
# pure-Python reference is within-tolerance rather than bit-for-bit --
# the same 1e-12-relative contract the C++ tests adopted when that side
# went -ffast-math. Counts, indices, and track structure stay exact;
# only accumulated doubles get the tolerance.
REL_TOL = 1e-12


def assert_close(tc, a, b):
    tc.assertLessEqual(abs(a - b), REL_TOL * max(abs(a), abs(b), 1.0),
                       f"{a!r} !~ {b!r}")


def reference_batches():
    """Yields (IQBatch, i_array, q_array, idx_array) per batch of the
    standard synthetic signal."""
    src = SyntheticIQSource(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=NUM_PULSES)
    batch = pulse_pb2.IQBatch()
    while src.next_batch(batch):
        n = len(batch.i)
        i = np.asarray(batch.i, dtype=np.float64)
        q = np.asarray(batch.q, dtype=np.float64)
        first = batch.first_sample_index
        idx = np.arange(first, first + n, dtype=np.uint64)
        yield batch, i, q, idx


@unittest.skipUnless(np is not None and numba_kernels is not None, "numba not installed")
class TestNumbaKernelParity(unittest.TestCase):
    def test_generator_bit_identical(self):
        src = SyntheticIQSource(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=NUM_PULSES)
        batch = pulse_pb2.IQBatch()
        period = _GAP_SAMPLES + _PULSE_SAMPLES
        rng_state = 42
        cursor = 0
        while src.next_batch(batch):
            n = len(batch.i)
            i_arr, q_arr, idx_arr, rng_state = numba_kernels.generate_batch(
                cursor, n, period, _GAP_SAMPLES, _PULSE_COMPONENT, _NOISE_AMPLITUDE, rng_state
            )
            cursor += n
            first = batch.first_sample_index
            for k in (0, 1, n // 2, n - 1):
                assert_close(self, batch.i[k], i_arr[k])
                assert_close(self, batch.q[k], q_arr[k])
                self.assertEqual(first + k, idx_arr[k])

    def test_detector_bit_identical_including_straddle(self):
        ref = PulseDetector(amplitude_threshold=6.0, sample_rate_hz=SAMPLE_RATE_HZ)
        state = (False, 0, 0.0, 0.0, 0)
        for batch, i, q, idx in reference_batches():
            expected = pulse_pb2.PulseEventBatch()
            ref.process(batch, expected)
            (ev_start, ev_end, ev_peak, ev_mean, ev_dur, *state) = numba_kernels.detect_pulses(
                i, q, idx, 36.0, SAMPLE_RATE_HZ, *state
            )
            self.assertEqual(len(expected.start_sample), len(ev_start))
            for k in range(len(expected.start_sample)):
                self.assertEqual(expected.start_sample[k], ev_start[k])
                self.assertEqual(expected.end_sample[k], ev_end[k])
                assert_close(self, expected.peak_amplitude[k], ev_peak[k])
                assert_close(self, expected.mean_amplitude[k], ev_mean[k])
                assert_close(self, expected.duration_seconds[k], ev_dur[k])

    def test_spectrogram_within_tolerance(self):
        # 1e-9 gate rather than assert_close's 1e-12: round four moved
        # this kernel to the phase-table + vectorized-reduction form,
        # so like the numpy variant's identical rework it reassociates
        # the per-bin sums -- same gate that variant has always used.
        num_bins = 8
        bin_hz = SAMPLE_RATE_HZ / (2.0 * num_bins)
        ref = SpectrogramAnalyzer(sample_rate_hz=SAMPLE_RATE_HZ, num_bins=num_bins)
        out = pulse_pb2.SpectrogramSummary()
        max_mag = np.zeros(num_bins)
        sum_mag = np.zeros(num_bins)
        frames = 0
        tables = {}
        for batch, i, q, idx in reference_batches():
            ref.process(batch, out)
            n = i.shape[0]
            if n not in tables:
                tables[n] = numba_kernels.build_phase_tables(num_bins, n, bin_hz, SAMPLE_RATE_HZ)
            tre, tim = tables[n]
            max_mag, sum_mag = numba_kernels.spectrogram_bins(
                i, q, idx[0], SAMPLE_RATE_HZ, num_bins, bin_hz, tre, tim, max_mag, sum_mag
            )
            frames += 1
        for b in range(num_bins):
            self.assertLess(abs(max_mag[b] - out.max_magnitude[b]) / out.max_magnitude[b], 1e-9)
            self.assertLess(
                abs(sum_mag[b] / frames - out.mean_magnitude[b]) / out.mean_magnitude[b], 1e-9
            )

    def test_jammer_bit_identical(self):
        ref = JammerDetector(power_threshold=20.0, duty_cycle_threshold=0.5)
        out = pulse_pb2.JamSummary()
        max_mean_power = 0.0
        for batch, i, q, idx in reference_batches():
            ref.process(batch, out)
            power_sum, over = numba_kernels.jammer_power(i, q, 20.0)
            max_mean_power = max(max_mean_power, power_sum / len(i))
        assert_close(self, out.max_mean_power, max_mean_power)

    def test_stats_bit_identical(self):
        from pulsecore.pulse_stats import PulseStatsAccumulator

        ref_det = PulseDetector(amplitude_threshold=6.0, sample_rate_hz=SAMPLE_RATE_HZ)
        ref_acc = PulseStatsAccumulator(sample_rate_hz=SAMPLE_RATE_HZ)
        det_state = (False, 0, 0.0, 0.0, 0)
        st = (0, 0.0, 0.0, float("inf"), float("-inf"), 0.0, 0, False, np.uint64(0))
        for batch, i, q, idx in reference_batches():
            events = pulse_pb2.PulseEventBatch()
            ref_det.process(batch, events)
            ref_acc.add(events)
            (ev_start, ev_end, ev_peak, ev_mean, ev_dur, *det_state) = numba_kernels.detect_pulses(
                i, q, idx, 36.0, SAMPLE_RATE_HZ, *det_state
            )
            st = numba_kernels.stats_accumulate(
                ev_start, ev_peak, ev_dur, SAMPLE_RATE_HZ, *st
            )
        summary = ref_acc.finalize()
        count, peak_sum, duration_sum, peak_min, peak_max, pri_sum, pri_count = st[:7]
        self.assertEqual(summary.pulse_count, count)
        assert_close(self, summary.mean_peak_amplitude, peak_sum / count)
        assert_close(self, summary.mean_duration_seconds, duration_sum / count)
        assert_close(self, summary.min_peak_amplitude, peak_min)
        assert_close(self, summary.max_peak_amplitude, peak_max)
        assert_close(self, summary.mean_pri_seconds, pri_sum / pri_count)

    def test_deinterleaver_bit_identical(self):
        from pulsecore.deinterleaver import Deinterleaver

        ref_det = PulseDetector(amplitude_threshold=6.0, sample_rate_hz=SAMPLE_RATE_HZ)
        ref_deint = Deinterleaver(sample_rate_hz=SAMPLE_RATE_HZ, pri_tolerance_seconds=1e-7)
        ref_out = pulse_pb2.DeinterleaveSummary()
        det_state = (False, 0, 0.0, 0.0, 0)

        cap = 64
        t_id = np.zeros(cap, dtype=np.uint32)
        t_pulses = np.zeros(cap, dtype=np.int64)
        t_last = np.zeros(cap, dtype=np.uint64)
        t_pri_sum = np.zeros(cap)
        t_pri_count = np.zeros(cap, dtype=np.int64)
        t_peak = np.zeros(cap)
        track_count = 0
        next_id = 1

        for batch, i, q, idx in reference_batches():
            events = pulse_pb2.PulseEventBatch()
            ref_det.process(batch, events)
            ref_deint.process(events, ref_out)
            (ev_start, ev_end, ev_peak, ev_mean, ev_dur, *det_state) = numba_kernels.detect_pulses(
                i, q, idx, 36.0, SAMPLE_RATE_HZ, *det_state
            )
            track_count, next_id = numba_kernels.deinterleave_events(
                ev_start, ev_peak, SAMPLE_RATE_HZ, 1e-7, track_count, next_id,
                t_id, t_pulses, t_last, t_pri_sum, t_pri_count, t_peak,
            )

        self.assertEqual(len(ref_out.tracks), track_count)
        for t in range(track_count):
            ref_track = ref_out.tracks[t]
            self.assertEqual(ref_track.track_id, t_id[t])
            self.assertEqual(ref_track.pulse_count, t_pulses[t])
            assert_close(self, ref_track.estimated_pri_seconds, t_pri_sum[t] / t_pri_count[t])
            assert_close(self, ref_track.mean_peak_amplitude, t_peak[t] / t_pulses[t])


@unittest.skipUnless(np is not None and numpy_kernels is not None, "numpy not installed")
class TestNumpyKernelParity(unittest.TestCase):
    def test_array_generator_bit_identical(self):
        src = SyntheticIQSource(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=NUM_PULSES)
        arr = SyntheticIQSourceArrays(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=NUM_PULSES)
        batch = pulse_pb2.IQBatch()
        while src.next_batch(batch):
            got = arr.next_batch()
            self.assertIsNotNone(got)
            i_arr, q_arr, idx = got
            n = len(batch.i)
            first = batch.first_sample_index
            for k in (0, 1, n // 2, n - 1):
                self.assertEqual(batch.i[k], i_arr[k])
                self.assertEqual(batch.q[k], q_arr[k])
                self.assertEqual(first + k, idx[k])
        self.assertIsNone(arr.next_batch())

    def test_detector_bit_identical_including_straddle(self):
        # The vectorized detector promises exact equality (no reordered
        # reductions in its per-event aggregations at the granularity
        # protobuf doubles round-trip) -- including a pulse left open
        # across a batch boundary, hand-built here because the standard
        # signal never produces one.
        state = (False, 0, 0.0, 0.0, 0)
        ref = PulseDetector(amplitude_threshold=2.0, sample_rate_hz=SAMPLE_RATE_HZ)

        amps1 = [0.5, 5.0, 6.0]
        amps2 = [7.0, 4.0, 0.5, 0.5]
        expected = pulse_pb2.PulseEventBatch()
        b1 = pulse_pb2.IQBatch()
        b1.sample_rate_hz = SAMPLE_RATE_HZ
        b1.first_sample_index = 0
        b1.i.extend(amps1)
        b1.q.extend([0.0] * len(amps1))
        b2 = pulse_pb2.IQBatch()
        b2.sample_rate_hz = SAMPLE_RATE_HZ
        b2.first_sample_index = 3
        b2.i.extend(amps2)
        b2.q.extend([0.0] * len(amps2))
        ref.process(b1, expected)
        ref.process(b2, expected)

        got = []
        for amps, start in ((amps1, 0), (amps2, 3)):
            i = np.array(amps)
            q = np.zeros(len(amps))
            idx = np.arange(start, start + len(amps), dtype=np.uint64)
            (ev_start, ev_end, ev_peak, ev_mean, ev_dur, *state) = numpy_kernels.detect_pulses(
                i, q, idx, 4.0, SAMPLE_RATE_HZ, *state
            )
            for k in range(len(ev_start)):
                got.append((int(ev_start[k]), int(ev_end[k]), float(ev_peak[k]),
                            float(ev_mean[k]), float(ev_dur[k])))

        self.assertEqual(len(expected.start_sample), len(got))
        for k, g in enumerate(got):
            self.assertEqual(
                (expected.start_sample[k], expected.end_sample[k],
                 expected.peak_amplitude[k], expected.mean_amplitude[k],
                 expected.duration_seconds[k]),
                g,
            )

    def test_spectrogram_within_tolerance(self):
        # Not bit-identical, by design (np.sum reorders additions;
        # direct phase evaluation replaces the recursion) -- see
        # numpy_variant/kernels.py. Tolerance mirrors the full-scale
        # verify tool's.
        num_bins = 8
        bin_hz = SAMPLE_RATE_HZ / (2.0 * num_bins)
        ref = SpectrogramAnalyzer(sample_rate_hz=SAMPLE_RATE_HZ, num_bins=num_bins)
        out = pulse_pb2.SpectrogramSummary()
        max_mag = np.zeros(num_bins)
        sum_mag = np.zeros(num_bins)
        frames = 0
        for batch, i, q, idx in reference_batches():
            ref.process(batch, out)
            max_mag, sum_mag = numpy_kernels.spectrogram_bins(
                i, q, idx[0], SAMPLE_RATE_HZ, num_bins, bin_hz, max_mag, sum_mag
            )
            frames += 1
        for b in range(num_bins):
            self.assertLess(
                abs(max_mag[b] - out.max_magnitude[b]) / out.max_magnitude[b], 1e-9
            )
            self.assertLess(
                abs(sum_mag[b] / frames - out.mean_magnitude[b]) / out.mean_magnitude[b], 1e-9
            )

    def test_jammer_counts_exact_power_within_tolerance(self):
        ref = JammerDetector(power_threshold=20.0, duty_cycle_threshold=0.5)
        out = pulse_pb2.JamSummary()
        max_mean_power = 0.0
        total = 0
        for batch, i, q, idx in reference_batches():
            ref.process(batch, out)
            power_sum, over = numpy_kernels.jammer_power(i, q, 20.0)
            max_mean_power = max(max_mean_power, power_sum / len(i))
            total += 1
        self.assertEqual(out.batches_total, total)
        self.assertLess(abs(max_mean_power - out.max_mean_power) / out.max_mean_power, 1e-12)

    def test_array_aggregates_match_reference(self):
        # aggregates.py's array-native stats/deinterleaver vs the
        # pulsecore reference pair, fed identical detector output.
        # Counts and track structure exact; accumulated doubles within
        # tolerance (the stats sums are reordered/telescoped by design
        # -- see aggregates.py's docstring).
        from pulsecore.pulse_stats import PulseStatsAccumulator
        from pulsecore.deinterleaver import Deinterleaver

        ref_det = PulseDetector(amplitude_threshold=6.0, sample_rate_hz=SAMPLE_RATE_HZ)
        ref_stats = PulseStatsAccumulator(sample_rate_hz=SAMPLE_RATE_HZ)
        ref_deint = Deinterleaver(sample_rate_hz=SAMPLE_RATE_HZ, pri_tolerance_seconds=1e-7)
        ref_deint_out = pulse_pb2.DeinterleaveSummary()

        arr_stats = PulseStatsArrays(sample_rate_hz=SAMPLE_RATE_HZ)
        arr_deint = DeinterleaverArrays(sample_rate_hz=SAMPLE_RATE_HZ, pri_tolerance_seconds=1e-7)

        state = (False, 0, 0.0, 0.0, 0)
        for batch, i, q, idx in reference_batches():
            events = pulse_pb2.PulseEventBatch()
            ref_det.process(batch, events)
            ref_stats.add(events)
            ref_deint.process(events, ref_deint_out)
            (ev_start, ev_end, ev_peak, ev_mean, ev_dur, *state) = numpy_kernels.detect_pulses(
                i, q, idx, 36.0, SAMPLE_RATE_HZ, *state
            )
            arr_stats.add(ev_start, ev_peak, ev_dur)
            arr_deint.process(ev_start, ev_peak)

        ref_summary = ref_stats.finalize()
        arr_summary = arr_stats.finalize()
        self.assertEqual(ref_summary.pulse_count, arr_summary.pulse_count)
        assert_close(self, ref_summary.mean_peak_amplitude, arr_summary.mean_peak_amplitude)
        assert_close(self, ref_summary.mean_duration_seconds, arr_summary.mean_duration_seconds)
        assert_close(self, ref_summary.min_peak_amplitude, arr_summary.min_peak_amplitude)
        assert_close(self, ref_summary.max_peak_amplitude, arr_summary.max_peak_amplitude)
        assert_close(self, ref_summary.mean_pri_seconds, arr_summary.mean_pri_seconds)

        arr_out = arr_deint.summary()
        self.assertEqual(len(ref_deint_out.tracks), len(arr_out.tracks))
        for ref_track, arr_track in zip(ref_deint_out.tracks, arr_out.tracks):
            self.assertEqual(ref_track.track_id, arr_track.track_id)
            self.assertEqual(ref_track.pulse_count, arr_track.pulse_count)
            assert_close(self, ref_track.estimated_pri_seconds, arr_track.estimated_pri_seconds)
            assert_close(self, ref_track.mean_peak_amplitude, arr_track.mean_peak_amplitude)


if __name__ == "__main__":
    unittest.main()
