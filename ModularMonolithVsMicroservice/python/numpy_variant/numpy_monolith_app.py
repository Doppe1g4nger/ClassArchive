#!/usr/bin/env python3
"""numpy_monolith_app.py: a fifth Python architecture. Same five-stage
pipeline as python/monolith/monolith_app.py, with detector, spectrogram,
and jammer rewritten as bulk numpy array operations (see kernels.py) and
IQ generation writing straight into numpy arrays (see
iq_source_arrays.py -- originally it reused pulsecore's protobuf-based
generator, until profiling measured that round-trip as this variant's
single biggest cost). pulse_stats and the deinterleaver are reused from
pulsecore as-is; see __init__.py for why they've stayed pure Python
here even after numba_variant jitted its own. See kernels.py's module
docstring for what's different about this variant's numbers versus
every other build in this repo (numerically equivalent, not
bit-identical).

    numpy_monolith_app.py [num_pulses]
"""
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from numpy_variant import kernels
from numpy_variant.aggregates import DeinterleaverArrays, PulseStatsArrays
from numpy_variant.iq_source_arrays import SyntheticIQSourceArrays

_SAMPLE_RATE_HZ = 10_000_000.0
_NUM_BINS = 8
_THRESHOLD = 6.0
_POWER_THRESHOLD = 20.0
_DUTY_CYCLE_THRESHOLD = 0.5
_PRI_TOLERANCE_SECONDS = 1e-7


def main() -> int:
    num_pulses = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    threshold_sq = _THRESHOLD * _THRESHOLD
    bin_hz = _SAMPLE_RATE_HZ / (2.0 * _NUM_BINS)

    # Generates straight into numpy arrays -- cProfile showed the old
    # protobuf-then-extract path (SyntheticIQSource + array_view) was
    # this variant's single biggest cost, bigger than all its vectorized
    # kernels combined. See iq_source_arrays.py.
    source = SyntheticIQSourceArrays(sample_rate_hz=_SAMPLE_RATE_HZ, num_pulses=num_pulses)
    # Round three: stats and the deinterleaver are array-native too
    # (see aggregates.py) -- no protobuf anywhere on the per-batch path.
    accumulator = PulseStatsArrays(sample_rate_hz=_SAMPLE_RATE_HZ)
    deinterleaver = DeinterleaverArrays(
        sample_rate_hz=_SAMPLE_RATE_HZ, pri_tolerance_seconds=_PRI_TOLERANCE_SECONDS
    )

    # Round three: the signal is a GIVEN (real IQ comes from a radio),
    # so every batch is generated before the clock starts and the
    # measured region begins at detection -- the same charter as every
    # other build on this branch.
    pregenerated = []
    while True:
        got = source.next_batch()
        if got is None:
            break
        pregenerated.append(got)

    batches = 0
    in_pulse = False
    pulse_start = 0
    pulse_peak = 0.0
    pulse_sum = 0.0
    pulse_sample_count = 0

    max_magnitude = np.zeros(_NUM_BINS)
    sum_magnitude = np.zeros(_NUM_BINS)
    frame_count = 0

    batches_total = 0
    batches_flagged = 0
    max_duty_cycle = 0.0
    max_mean_power = 0.0

    # Timed region covers detection through deinterleave over the
    # pre-generated batches, same convention as every other build.
    steady_state_start = time.perf_counter()
    for i_arr, q_arr, idx_arr in pregenerated:
        n = i_arr.shape[0]

        (ev_start, ev_end, ev_peak, ev_mean, ev_dur, in_pulse, pulse_start, pulse_peak,
         pulse_sum, pulse_sample_count) = kernels.detect_pulses(
            i_arr, q_arr, idx_arr, threshold_sq, _SAMPLE_RATE_HZ,
            in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_sample_count,
        )

        max_magnitude, sum_magnitude = kernels.spectrogram_bins(
            i_arr, q_arr, idx_arr[0], _SAMPLE_RATE_HZ, _NUM_BINS, bin_hz,
            max_magnitude, sum_magnitude,
        )
        frame_count += 1

        power_sum, over_threshold = kernels.jammer_power(i_arr, q_arr, _POWER_THRESHOLD)
        mean_power = power_sum / n
        duty_cycle = over_threshold / n
        batches_total += 1
        if duty_cycle >= _DUTY_CYCLE_THRESHOLD:
            batches_flagged += 1
        if duty_cycle > max_duty_cycle:
            max_duty_cycle = duty_cycle
        if mean_power > max_mean_power:
            max_mean_power = mean_power

        # Event arrays flow straight into the array-native aggregates
        # (aggregates.py) -- the per-batch protobuf event round-trip
        # that used to live here is gone.
        accumulator.add(ev_start, ev_peak, ev_dur)
        deinterleaver.process(ev_start, ev_peak)

        batches += 1
    steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0
    deinterleave_summary = deinterleaver.summary()

    print(
        f"[numpy_monolith_app.py] processed {batches} IQ batches through 4 vectorized stages "
        f"+ 1 array-native deinterleaver"
    )
    print(f"[numpy_monolith_app.py] STEADY_STATE_MS {steady_state_ms:.6f}")

    print(
        f"[numpy_monolith_app.py] spectrogram: {_NUM_BINS} bins, {bin_hz:.1f} Hz spacing, "
        f"{frame_count} frames"
    )
    for i in range(_NUM_BINS):
        bin_center = (i + 0.5) * bin_hz
        mean = sum_magnitude[i] / frame_count if frame_count > 0 else 0.0
        print(
            f"[numpy_monolith_app.py]   bin {i} (~{bin_center:.0f} Hz): "
            f"max={max_magnitude[i]:.3f} mean={mean:.3f}"
        )

    print(
        f"[numpy_monolith_app.py] jammer: {batches_flagged}/{batches_total} batches flagged, "
        f"max_duty_cycle={max_duty_cycle:.3f} max_mean_power={max_mean_power:.2f}"
    )

    stats = accumulator.finalize()
    print(
        f"[numpy_monolith_app.py] stats: pulses={stats.pulse_count} "
        f"mean_peak={stats.mean_peak_amplitude:.3f} "
        f"mean_dur_us={stats.mean_duration_seconds * 1e6:.2f} "
        f"mean_pri_us={stats.mean_pri_seconds * 1e6:.2f} "
        f"min_peak={stats.min_peak_amplitude:.3f} max_peak={stats.max_peak_amplitude:.3f}"
    )

    print(f"[numpy_monolith_app.py] deinterleaver: {len(deinterleave_summary.tracks)} track(s)")
    for track in deinterleave_summary.tracks:
        print(
            f"[numpy_monolith_app.py]   track {track.track_id}: pulses={track.pulse_count} "
            f"estimated_pri_us={track.estimated_pri_seconds * 1e6:.2f} "
            f"mean_peak={track.mean_peak_amplitude:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
