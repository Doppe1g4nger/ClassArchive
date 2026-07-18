#!/usr/bin/env python3
"""Standalone correctness check for numpy_variant/kernels.py, run
separately from the usual output-diff checks this repo uses everywhere
else, because this variant's output is *not* expected to be bit-
identical to the rest of the repo (see kernels.py's module docstring for
why). This script exists to make that claim checkable instead of just
asserted: it runs the real synthetic IQ source through both the scalar
reference (pulsecore.spectrogram/jammer) and the vectorized kernels,
batch by batch, and reports the actual measured relative error -- along
with an exact-match check for the detector, whose output *is* expected
to be bit-identical (see detect_pulses' docstring for why the detector
doesn't have the same floating-point-reordering exposure the other two
kernels do), and a dedicated hand-built test for a pulse straddling a
batch boundary, a case this repo's own synthetic signal never actually
exercises (its batch size is an exact multiple of the pulse period) but
the vectorized detector still has to handle correctly.

    verify_numpy_variant.py [num_pulses]
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.iq_source import SyntheticIQSource
from pulsecore.pulse_detector import PulseDetector
from pulsecore.spectrogram import SpectrogramAnalyzer
from pulsecore.jammer import JammerDetector
from pulsecore.array_view import extract_iq
from numpy_variant import kernels

_SAMPLE_RATE_HZ = 10_000_000.0
_NUM_BINS = 8
_THRESHOLD = 6.0
_POWER_THRESHOLD = 20.0


def check_straddling() -> bool:
    """A pulse that starts in one batch and closes in the next -- see
    the module docstring for why this repo's own signal never produces
    one naturally at its current batch-size/period constants."""
    threshold_sq = _THRESHOLD * _THRESHOLD

    def make_batch(amplitudes, start_idx):
        b = pulse_pb2.IQBatch()
        b.sample_rate_hz = _SAMPLE_RATE_HZ
        for k, amp in enumerate(amplitudes):
            s = b.samples.add()
            s.sample_index = start_idx + k
            s.i = amp
            s.q = 0.0
        return b

    batch1_amps = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 20.0, 21.0, 22.0]
    batch2_amps = [23.0, 24.0, 0.5, 0.5, 30.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    batch1 = make_batch(batch1_amps, 0)
    batch2 = make_batch(batch2_amps, 10)

    ref = PulseDetector(_THRESHOLD, _SAMPLE_RATE_HZ)
    ref_events = pulse_pb2.PulseEventBatch()
    ref.process(batch1, ref_events)
    ref.process(batch2, ref_events)
    ref_result = [
        (e.start_sample, e.end_sample, e.peak_amplitude, e.mean_amplitude, e.duration_seconds)
        for e in ref_events.events
    ]

    i1, q1, idx1 = np.array(batch1_amps), np.zeros(10), np.arange(0, 10, dtype=np.uint64)
    i2, q2, idx2 = np.array(batch2_amps), np.zeros(10), np.arange(10, 20, dtype=np.uint64)
    state = (False, 0, 0.0, 0.0, 0)
    ev1 = kernels.detect_pulses(i1, q1, idx1, threshold_sq, _SAMPLE_RATE_HZ, *state)
    state = ev1[5:]
    ev2 = kernels.detect_pulses(i2, q2, idx2, threshold_sq, _SAMPLE_RATE_HZ, *state)

    got_result = []
    for ev in (ev1, ev2):
        starts, ends, peaks, means, durs = ev[:5]
        for k in range(len(starts)):
            got_result.append((int(starts[k]), int(ends[k]), float(peaks[k]), float(means[k]), float(durs[k])))

    ok = got_result == ref_result
    print(f"straddling test: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  reference:", ref_result)
        print("  got:      ", got_result)
    return ok


def check_against_real_signal(num_pulses: int) -> bool:
    threshold_sq = _THRESHOLD * _THRESHOLD
    bin_hz = _SAMPLE_RATE_HZ / (2.0 * _NUM_BINS)

    source = SyntheticIQSource(sample_rate_hz=_SAMPLE_RATE_HZ, num_pulses=num_pulses)
    ref_detector = PulseDetector(_THRESHOLD, _SAMPLE_RATE_HZ)
    ref_spec = SpectrogramAnalyzer(_SAMPLE_RATE_HZ, _NUM_BINS)
    ref_jam = JammerDetector(_POWER_THRESHOLD, 0.5)
    ref_events = pulse_pb2.PulseEventBatch()
    ref_spec_out = pulse_pb2.SpectrogramSummary()
    ref_jam_out = pulse_pb2.JamSummary()

    in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_count = False, 0, 0.0, 0.0, 0
    max_magnitude = np.zeros(_NUM_BINS)
    sum_magnitude = np.zeros(_NUM_BINS)
    max_mean_power = 0.0

    iq_batch = pulse_pb2.IQBatch()
    detector_exact = True
    max_rel_err = 0.0

    while source.next_batch(iq_batch):
        ref_events.Clear()
        ref_detector.process(iq_batch, ref_events)
        ref_spec.process(iq_batch, ref_spec_out)
        ref_jam.process(iq_batch, ref_jam_out)

        i_arr, q_arr, idx_arr = extract_iq(iq_batch)
        n = i_arr.shape[0]

        (ev_start, ev_end, ev_peak, ev_mean, ev_dur, in_pulse, pulse_start, pulse_peak,
         pulse_sum, pulse_count) = kernels.detect_pulses(
            i_arr, q_arr, idx_arr, threshold_sq, _SAMPLE_RATE_HZ,
            in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_count,
        )
        if len(ref_events.events) != len(ev_start):
            detector_exact = False
        else:
            for k, e in enumerate(ref_events.events):
                if (e.start_sample != ev_start[k] or e.end_sample != ev_end[k]
                        or e.peak_amplitude != ev_peak[k] or e.mean_amplitude != ev_mean[k]
                        or e.duration_seconds != ev_dur[k]):
                    detector_exact = False

        max_magnitude, sum_magnitude = kernels.spectrogram_bins(
            i_arr, q_arr, idx_arr[0], _SAMPLE_RATE_HZ, _NUM_BINS, bin_hz,
            max_magnitude, sum_magnitude,
        )
        power_sum, _ = kernels.jammer_power(i_arr, q_arr, _POWER_THRESHOLD)
        max_mean_power = max(max_mean_power, power_sum / n)

    ref_max = np.array(ref_spec_out.max_magnitude)
    ref_mean = np.array(ref_spec_out.mean_magnitude)
    got_mean = sum_magnitude / ref_spec_out.frame_count
    rel_err_max = np.max(np.abs(max_magnitude - ref_max) / np.abs(ref_max))
    rel_err_mean = np.max(np.abs(got_mean - ref_mean) / np.abs(ref_mean))
    rel_err_power = abs(max_mean_power - ref_jam_out.max_mean_power) / ref_jam_out.max_mean_power
    max_rel_err = max(rel_err_max, rel_err_mean, rel_err_power)

    print(f"detector (exact match required): {'OK' if detector_exact else 'MISMATCH'}")
    print(f"spectrogram max_magnitude relative error: {rel_err_max:.3e}")
    print(f"spectrogram mean_magnitude relative error: {rel_err_mean:.3e}")
    print(f"jammer max_mean_power relative error: {rel_err_power:.3e}")
    print(f"worst relative error observed: {max_rel_err:.3e} (tolerance: 1e-9)")

    return detector_exact and max_rel_err < 1e-9


def main() -> int:
    num_pulses = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    ok_straddle = check_straddling()
    print()
    ok_signal = check_against_real_signal(num_pulses)
    print()
    print("PASS" if (ok_straddle and ok_signal) else "FAIL")
    return 0 if (ok_straddle and ok_signal) else 1


if __name__ == "__main__":
    sys.exit(main())
