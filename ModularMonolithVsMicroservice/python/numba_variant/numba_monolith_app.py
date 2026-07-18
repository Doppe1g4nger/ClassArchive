#!/usr/bin/env python3
"""numba_monolith_app.py: a fourth Python architecture. Runs the exact
same five-algorithm pipeline as python/monolith/monolith_app.py, with
every piece of per-batch work -- IQ generation, detector, spectrogram,
jammer, stats, deinterleaver -- @njit-compiled (see kernels.py)
operating on flat numpy arrays. The stats accumulator and deinterleaver
started out as reused pure-Python pulsecore code on the theory that
they'd never matter (they see ~1,000 events/batch, not 10,000 samples);
profiling then showed that with everything else compiled they were
essentially all of the remaining steady-state time, so a second pass
jitted them too and dropped the protobuf event rebuild that existed
only to feed them.

Unlike monolith/stages.py's uniform Stage.process(frame) interface,
nothing here touches pulse_pb2 in the hot path and there's no per-stage
abstraction either -- both exist in the other builds to cross a
boundary (a dlopen() call, a process, a wire) that a single-process
compiled build doesn't have. That's a deliberate departure from the
"modular" story the other builds tell, not an oversight: this variant
exists to answer a performance question (how fast can Python get once
the hot loops are compiled), not to demonstrate a plugin architecture.

    numba_monolith_app.py [num_pulses]
"""
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore.iq_source import (
    _BATCH_SIZE,
    _GAP_SAMPLES,
    _PULSE_SAMPLES,
    _PULSE_COMPONENT,
    _NOISE_AMPLITUDE,
)
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
    ev = np.array([0, 10], dtype=np.uint64)
    pk = np.array([1.0, 2.0])
    du = np.array([1e-7, 1e-7])
    kernels.stats_accumulate(ev, pk, du, _SAMPLE_RATE_HZ, 0, 0.0, 0.0,
                              float("inf"), float("-inf"), 0.0, 0, False, np.uint64(0))
    kernels.deinterleave_events(ev, pk, _SAMPLE_RATE_HZ, _PRI_TOLERANCE_SECONDS, 0, 1,
                                 np.zeros(4, dtype=np.uint32), np.zeros(4, dtype=np.int64),
                                 np.zeros(4, dtype=np.uint64), np.zeros(4),
                                 np.zeros(4, dtype=np.int64), np.zeros(4))


def main() -> int:
    num_pulses = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    _warm_up()

    period = _GAP_SAMPLES + _PULSE_SAMPLES
    total_samples = num_pulses * period + _GAP_SAMPLES
    bin_hz = _SAMPLE_RATE_HZ / (2.0 * _NUM_BINS)
    threshold_sq = _THRESHOLD * _THRESHOLD

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

    # Stats accumulator state, carried across batches -- same fields,
    # same initial values as pulse_stats.PulseStatsAccumulator, now fed
    # to the jitted stats_accumulate kernel (see kernels.py for why the
    # last two stages got jitted in a second pass).
    st_count = 0
    st_peak_sum = 0.0
    st_duration_sum = 0.0
    st_peak_min = float("inf")
    st_peak_max = float("-inf")
    st_pri_sum = 0.0
    st_pri_count = 0
    st_have_prev = False
    st_prev_start = np.uint64(0)

    # Deinterleaver track state as parallel arrays for the jitted
    # deinterleave_events kernel. Grown ahead of each call to the worst
    # case (every event starts a new track) so the kernel never needs to
    # reallocate.
    track_capacity = 16
    track_id = np.zeros(track_capacity, dtype=np.uint32)
    track_pulse_count = np.zeros(track_capacity, dtype=np.int64)
    track_last_start = np.zeros(track_capacity, dtype=np.uint64)
    track_pri_sum = np.zeros(track_capacity)
    track_pri_count = np.zeros(track_capacity, dtype=np.int64)
    track_peak_sum = np.zeros(track_capacity)
    track_count = 0
    next_track_id = 1

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

        # Stats and deinterleaving consume the detector kernel's output
        # arrays directly -- no protobuf anywhere in this loop. (An
        # earlier version rebuilt a PulseEventBatch here purely to feed
        # the pure-Python stats/deinterleaver; cProfile showed that
        # rebuild plus those two interpreted stages were nearly all of
        # this build's remaining steady-state time, so they became
        # kernels too -- see kernels.py.)
        (st_count, st_peak_sum, st_duration_sum, st_peak_min, st_peak_max,
         st_pri_sum, st_pri_count, st_have_prev, st_prev_start) = kernels.stats_accumulate(
            ev_start, ev_peak, ev_dur, _SAMPLE_RATE_HZ,
            st_count, st_peak_sum, st_duration_sum, st_peak_min, st_peak_max,
            st_pri_sum, st_pri_count, st_have_prev, st_prev_start,
        )

        # Worst case, every event starts a new track -- grow the state
        # arrays up front so the kernel never has to. (At this repo's
        # single-emitter scale track_count stays 1, so this never fires
        # after the first sizing; it's here so the kernel stays correct
        # for arbitrary inputs, same as the pure-Python version was.)
        needed = track_count + len(ev_start)
        if needed > track_capacity:
            while track_capacity < needed:
                track_capacity *= 2

            def grow(arr):
                grown = np.zeros(track_capacity, dtype=arr.dtype)
                grown[: arr.shape[0]] = arr
                return grown

            track_id = grow(track_id)
            track_pulse_count = grow(track_pulse_count)
            track_last_start = grow(track_last_start)
            track_pri_sum = grow(track_pri_sum)
            track_pri_count = grow(track_pri_count)
            track_peak_sum = grow(track_peak_sum)
        track_count, next_track_id = kernels.deinterleave_events(
            ev_start, ev_peak, _SAMPLE_RATE_HZ, _PRI_TOLERANCE_SECONDS,
            track_count, next_track_id,
            track_id, track_pulse_count, track_last_start,
            track_pri_sum, track_pri_count, track_peak_sum,
        )

        cursor += count
        batches += 1
    steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    print(
        f"[numba_monolith_app.py] processed {batches} IQ batches through 6 numba-jitted "
        f"kernels"
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

    # Same derived fields, same expressions, as PulseStatsAccumulator
    # .finalize() -- just computed from the kernel-carried state.
    mean_peak = st_peak_sum / st_count if st_count > 0 else 0.0
    mean_dur = st_duration_sum / st_count if st_count > 0 else 0.0
    mean_pri = st_pri_sum / st_pri_count if st_pri_count > 0 else 0.0
    min_peak = st_peak_min if st_count > 0 else 0.0
    max_peak = st_peak_max if st_count > 0 else 0.0
    print(
        f"[numba_monolith_app.py] stats: pulses={st_count} "
        f"mean_peak={mean_peak:.3f} "
        f"mean_dur_us={mean_dur * 1e6:.2f} "
        f"mean_pri_us={mean_pri * 1e6:.2f} "
        f"min_peak={min_peak:.3f} max_peak={max_peak:.3f}"
    )

    print(f"[numba_monolith_app.py] deinterleaver: {track_count} track(s)")
    for t in range(track_count):
        est_pri = track_pri_sum[t] / track_pri_count[t] if track_pri_count[t] > 0 else 0.0
        mean_pk = track_peak_sum[t] / track_pulse_count[t] if track_pulse_count[t] > 0 else 0.0
        print(
            f"[numba_monolith_app.py]   track {track_id[t]}: pulses={track_pulse_count[t]} "
            f"estimated_pri_us={est_pri * 1e6:.2f} "
            f"mean_peak={mean_pk:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
