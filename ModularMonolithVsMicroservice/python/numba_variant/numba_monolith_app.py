#!/usr/bin/env python3
"""numba_monolith_app.py: a fourth Python architecture. Runs the exact
same five-algorithm pipeline as python/monolith/monolith_app.py, but
with the three stages that touch the full 10,000-sample IQ batch
(detector, spectrogram, jammer) -- plus IQ generation itself -- replaced
by @njit-compiled kernels (see kernels.py) operating on flat numpy
arrays instead of pulse_pb2 messages. pulse_stats and the deinterleaver
are unchanged: they run over the much smaller `events` list, never the
bottleneck this variant targets, so JIT-compiling them would add
complexity for no measurable win.

Unlike monolith/stages.py's uniform Stage.process(frame) interface, this
app generates each batch directly into numpy arrays (kernels.generate_batch)
and passes those same arrays to all three kernels -- there's no
pulse_pb2.IQBatch anywhere in the hot path, and no per-stage abstraction
either, because both exist in the other builds to cross a boundary (a
dlopen() call, a process, a wire) that a single-process vectorized build
doesn't have. Only the detected events get written into a
pulse_pb2.PulseEventBatch, since pulse_stats.py and deinterleaver.py are
reused as-is. That's a deliberate departure from the "modular" story the
other builds tell, not an oversight: this variant exists to answer a
performance question (how fast can Python get once the hot loops are
compiled), not to demonstrate a plugin architecture.

    numba_monolith_app.py [num_pulses]
"""
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.iq_source import (
    _BATCH_SIZE,
    _GAP_SAMPLES,
    _PULSE_SAMPLES,
    _PULSE_COMPONENT,
    _NOISE_AMPLITUDE,
)
from pulsecore.pulse_stats import PulseStatsAccumulator
from pulsecore.deinterleaver import Deinterleaver
from numba_variant import kernels

_SAMPLE_RATE_HZ = 10_000_000.0
_NUM_BINS = 8
_THRESHOLD = 6.0
_POWER_THRESHOLD = 20.0
_DUTY_CYCLE_THRESHOLD = 0.5
_PRI_TOLERANCE_SECONDS = 1e-7


def _warm_up():
    """Forces numba to JIT-compile (or load from its on-disk cache) every
    kernel before the timed region starts, the same way monolith_app.py
    excludes dlopen() and monolith_app.py (Python) excludes import: a
    one-time cost that has nothing to do with steady-state throughput.
    Dummy sizes are small (4 samples) purely so compilation is fast;
    correctness of the compiled code is verified separately, not here."""
    i = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([1.0, 2.0, 3.0, 4.0])
    idx = np.array([0, 1, 2, 3], dtype=np.uint64)
    kernels.generate_batch(0, 4, 10, 8, _PULSE_COMPONENT, _NOISE_AMPLITUDE, 42)
    kernels.detect_pulses(i, q, idx, 36.0, _SAMPLE_RATE_HZ, False, 0, 0.0, 0.0, 0)
    kernels.spectrogram_bins(i, q, 0, _SAMPLE_RATE_HZ, _NUM_BINS, 1.0,
                              np.zeros(_NUM_BINS), np.zeros(_NUM_BINS))
    kernels.jammer_power(i, q, _POWER_THRESHOLD)


def main() -> int:
    num_pulses = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    _warm_up()

    period = _GAP_SAMPLES + _PULSE_SAMPLES
    total_samples = num_pulses * period + _GAP_SAMPLES
    bin_hz = _SAMPLE_RATE_HZ / (2.0 * _NUM_BINS)
    threshold_sq = _THRESHOLD * _THRESHOLD

    accumulator = PulseStatsAccumulator(sample_rate_hz=_SAMPLE_RATE_HZ)
    deinterleaver = Deinterleaver(
        sample_rate_hz=_SAMPLE_RATE_HZ, pri_tolerance_seconds=_PRI_TOLERANCE_SECONDS
    )
    events_batch = pulse_pb2.PulseEventBatch()
    deinterleave_summary = pulse_pb2.DeinterleaveSummary()

    rng_state = 42
    cursor = 0
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

    # Timed region covers only the batch-processing loop, same convention
    # as every other build in this repo -- JIT warm-up above and imports
    # are excluded, so this reflects steady-state throughput.
    steady_state_start = time.perf_counter()
    while cursor < total_samples:
        count = min(_BATCH_SIZE, total_samples - cursor)

        i_arr, q_arr, idx_arr, rng_state = kernels.generate_batch(
            cursor, count, period, _GAP_SAMPLES, _PULSE_COMPONENT, _NOISE_AMPLITUDE, rng_state
        )

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
        mean_power = power_sum / count
        duty_cycle = over_threshold / count
        batches_total += 1
        if duty_cycle >= _DUTY_CYCLE_THRESHOLD:
            batches_flagged += 1
        if duty_cycle > max_duty_cycle:
            max_duty_cycle = duty_cycle
        if mean_power > max_mean_power:
            max_mean_power = mean_power

        # detect_pulses returns plain numpy arrays, not a pulse_pb2
        # message -- stats/deinterleaver still speak protobuf, so the
        # events this batch found are written into one exactly once
        # here, the same event count as the scalar builds (~1,000/batch
        # at this repo's scale), not the full 10,000-sample batch.
        events_batch.Clear()
        for k in range(len(ev_start)):
            e = events_batch.events.add()
            e.start_sample = int(ev_start[k])
            e.end_sample = int(ev_end[k])
            e.peak_amplitude = float(ev_peak[k])
            e.mean_amplitude = float(ev_mean[k])
            e.duration_seconds = float(ev_dur[k])

        accumulator.add(events_batch)
        deinterleaver.process(events_batch, deinterleave_summary)

        cursor += count
        batches += 1
    steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    print(
        f"[numba_monolith_app.py] processed {batches} IQ batches through {4} numba-jitted "
        f"kernels + 2 pure-Python stages"
    )
    print(f"[numba_monolith_app.py] STEADY_STATE_MS {steady_state_ms:.6f}")

    print(
        f"[numba_monolith_app.py] spectrogram: {_NUM_BINS} bins, {bin_hz:.1f} Hz spacing, "
        f"{frame_count} frames"
    )
    for i in range(_NUM_BINS):
        bin_center = (i + 0.5) * bin_hz
        mean = sum_magnitude[i] / frame_count if frame_count > 0 else 0.0
        print(
            f"[numba_monolith_app.py]   bin {i} (~{bin_center:.0f} Hz): "
            f"max={max_magnitude[i]:.3f} mean={mean:.3f}"
        )

    print(
        f"[numba_monolith_app.py] jammer: {batches_flagged}/{batches_total} batches flagged, "
        f"max_duty_cycle={max_duty_cycle:.3f} max_mean_power={max_mean_power:.2f}"
    )

    stats = accumulator.finalize()
    print(
        f"[numba_monolith_app.py] stats: pulses={stats.pulse_count} "
        f"mean_peak={stats.mean_peak_amplitude:.3f} "
        f"mean_dur_us={stats.mean_duration_seconds * 1e6:.2f} "
        f"mean_pri_us={stats.mean_pri_seconds * 1e6:.2f} "
        f"min_peak={stats.min_peak_amplitude:.3f} max_peak={stats.max_peak_amplitude:.3f}"
    )

    print(f"[numba_monolith_app.py] deinterleaver: {len(deinterleave_summary.tracks)} track(s)")
    for track in deinterleave_summary.tracks:
        print(
            f"[numba_monolith_app.py]   track {track.track_id}: pulses={track.pulse_count} "
            f"estimated_pri_us={track.estimated_pri_seconds * 1e6:.2f} "
            f"mean_peak={track.mean_peak_amplitude:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
