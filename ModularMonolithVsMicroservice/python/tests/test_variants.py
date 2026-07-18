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
    from numpy_variant.iq_source_arrays import SyntheticIQSourceArrays
except ImportError:  # pragma: no cover
    numpy_kernels = None

SAMPLE_RATE_HZ = 10_000_000.0
NUM_PULSES = 3000  # a few full batches, small enough for unit-test time


def reference_batches():
    """Yields (IQBatch, i_array, q_array, idx_array) per batch of the
    standard synthetic signal."""
    src = SyntheticIQSource(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=NUM_PULSES)
    batch = pulse_pb2.IQBatch()
    while src.next_batch(batch):
        n = len(batch.samples)
        i = np.fromiter((s.i for s in batch.samples), dtype=np.float64, count=n)
        q = np.fromiter((s.q for s in batch.samples), dtype=np.float64, count=n)
        idx = np.fromiter((s.sample_index for s in batch.samples), dtype=np.uint64, count=n)
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
            n = len(batch.samples)
            i_arr, q_arr, idx_arr, rng_state = numba_kernels.generate_batch(
                cursor, n, period, _GAP_SAMPLES, _PULSE_COMPONENT, _NOISE_AMPLITUDE, rng_state
            )
            cursor += n
            for k in (0, 1, n // 2, n - 1):
                self.assertEqual(batch.samples[k].i, i_arr[k])
                self.assertEqual(batch.samples[k].q, q_arr[k])
                self.assertEqual(batch.samples[k].sample_index, idx_arr[k])

    def test_detector_bit_identical_including_straddle(self):
        ref = PulseDetector(amplitude_threshold=6.0, sample_rate_hz=SAMPLE_RATE_HZ)
        state = (False, 0, 0.0, 0.0, 0)
        for batch, i, q, idx in reference_batches():
            expected = pulse_pb2.PulseEventBatch()
            ref.process(batch, expected)
            (ev_start, ev_end, ev_peak, ev_mean, ev_dur, *state) = numba_kernels.detect_pulses(
                i, q, idx, 36.0, SAMPLE_RATE_HZ, *state
            )
            self.assertEqual(len(expected.events), len(ev_start))
            for k, e in enumerate(expected.events):
                self.assertEqual(e.start_sample, ev_start[k])
                self.assertEqual(e.end_sample, ev_end[k])
                self.assertEqual(e.peak_amplitude, ev_peak[k])
                self.assertEqual(e.mean_amplitude, ev_mean[k])
                self.assertEqual(e.duration_seconds, ev_dur[k])

    def test_spectrogram_bit_identical(self):
        num_bins = 8
        bin_hz = SAMPLE_RATE_HZ / (2.0 * num_bins)
        ref = SpectrogramAnalyzer(sample_rate_hz=SAMPLE_RATE_HZ, num_bins=num_bins)
        out = pulse_pb2.SpectrogramSummary()
        max_mag = np.zeros(num_bins)
        sum_mag = np.zeros(num_bins)
        frames = 0
        for batch, i, q, idx in reference_batches():
            ref.process(batch, out)
            max_mag, sum_mag = numba_kernels.spectrogram_bins(
                i, q, idx[0], SAMPLE_RATE_HZ, num_bins, bin_hz, max_mag, sum_mag
            )
            frames += 1
        for b in range(num_bins):
            self.assertEqual(out.max_magnitude[b], max_mag[b])
            self.assertEqual(out.mean_magnitude[b], sum_mag[b] / frames)

    def test_jammer_bit_identical(self):
        ref = JammerDetector(power_threshold=20.0, duty_cycle_threshold=0.5)
        out = pulse_pb2.JamSummary()
        max_mean_power = 0.0
        for batch, i, q, idx in reference_batches():
            ref.process(batch, out)
            power_sum, over = numba_kernels.jammer_power(i, q, 20.0)
            max_mean_power = max(max_mean_power, power_sum / len(i))
        self.assertEqual(out.max_mean_power, max_mean_power)

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
        self.assertEqual(summary.mean_peak_amplitude, peak_sum / count)
        self.assertEqual(summary.mean_duration_seconds, duration_sum / count)
        self.assertEqual(summary.min_peak_amplitude, peak_min)
        self.assertEqual(summary.max_peak_amplitude, peak_max)
        self.assertEqual(summary.mean_pri_seconds, pri_sum / pri_count)

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
            self.assertEqual(ref_track.estimated_pri_seconds, t_pri_sum[t] / t_pri_count[t])
            self.assertEqual(ref_track.mean_peak_amplitude, t_peak[t] / t_pulses[t])


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
            n = len(batch.samples)
            for k in (0, 1, n // 2, n - 1):
                self.assertEqual(batch.samples[k].i, i_arr[k])
                self.assertEqual(batch.samples[k].q, q_arr[k])
                self.assertEqual(batch.samples[k].sample_index, idx[k])
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
        for k, a in enumerate(amps1):
            b1.samples.add(sample_index=k, i=a, q=0.0)
        b2 = pulse_pb2.IQBatch()
        b2.sample_rate_hz = SAMPLE_RATE_HZ
        for k, a in enumerate(amps2):
            b2.samples.add(sample_index=3 + k, i=a, q=0.0)
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

        self.assertEqual(len(expected.events), len(got))
        for e, g in zip(expected.events, got):
            self.assertEqual(
                (e.start_sample, e.end_sample, e.peak_amplitude, e.mean_amplitude,
                 e.duration_seconds),
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


if __name__ == "__main__":
    unittest.main()
