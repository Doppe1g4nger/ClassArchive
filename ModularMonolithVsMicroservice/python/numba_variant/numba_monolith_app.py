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
    """Forces numba to JIT-compile (or load from its on-disk cache) the
    fused pipeline kernel before the timed region starts, the same way
    monolith_app.py excludes dlopen() and the Python monolith excludes
    import: a one-time cost that has nothing to do with steady-state
    throughput. The dummy run is tiny (two 4-sample batches) purely so
    compilation is fast; correctness of the compiled code is verified
    separately (tests/test_variants.py), not here."""
    i = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    q = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
    idx = np.arange(8, dtype=np.uint64)
    offsets = np.array([0, 4, 8], dtype=np.int64)
    kernels.generate_batch(0, 4, 10, 8, _PULSE_COMPONENT, _NOISE_AMPLITUDE, 42)
    kernels.run_pipeline(
        i, q, idx, offsets, _SAMPLE_RATE_HZ, _NUM_BINS, 1.0, 36.0,
        _POWER_THRESHOLD, _DUTY_CYCLE_THRESHOLD, _PRI_TOLERANCE_SECONDS,
        np.zeros(_NUM_BINS), np.zeros(_NUM_BINS),
        np.zeros(16, dtype=np.uint32), np.zeros(16, dtype=np.int64),
        np.zeros(16, dtype=np.uint64), np.zeros(16),
        np.zeros(16, dtype=np.int64), np.zeros(16))


def main() -> int:
    num_pulses = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    _warm_up()

    period = _GAP_SAMPLES + _PULSE_SAMPLES
    total_samples = num_pulses * period + _GAP_SAMPLES
    bin_hz = _SAMPLE_RATE_HZ / (2.0 * _NUM_BINS)
    threshold_sq = _THRESHOLD * _THRESHOLD

    # Round three: the signal is a GIVEN (real IQ comes from a radio),
    # so the whole signal is generated before the clock starts -- one
    # generate_batch call for all of it (the generator's RNG and pulse
    # phase flow continuously, so one call produces the same samples
    # batching did) -- and the measured region begins at detection.
    # offsets[b] marks where batch b starts in the flat arrays,
    # preserving the exact batch boundaries every other build uses.
    i_all, q_all, idx_all, _ = kernels.generate_batch(
        0, total_samples, period, _GAP_SAMPLES, _PULSE_COMPONENT, _NOISE_AMPLITUDE, 42
    )
    offsets = np.arange(0, total_samples, _BATCH_SIZE, dtype=np.int64)
    offsets = np.append(offsets, np.int64(total_samples))

    max_magnitude = np.zeros(_NUM_BINS)
    sum_magnitude = np.zeros(_NUM_BINS)

    # Deinterleaver track state as parallel arrays. Capacity is a
    # caller contract now that the batch loop lives inside the fused
    # kernel (no per-batch growth point anymore): sized to this
    # signal's true event bound -- at most one event per pulse period,
    # since detection threshold 6.0 sits far above the noise floor --
    # plus slack. run_pipeline aborts with batches_total == -1 rather
    # than overflow if an input ever exceeds it.
    track_capacity = num_pulses + 16
    track_id = np.zeros(track_capacity, dtype=np.uint32)
    track_pulse_count = np.zeros(track_capacity, dtype=np.int64)
    track_last_start = np.zeros(track_capacity, dtype=np.uint64)
    track_pri_sum = np.zeros(track_capacity)
    track_pri_count = np.zeros(track_capacity, dtype=np.int64)
    track_peak_sum = np.zeros(track_capacity)

    # Timed region is ONE call: the entire detection-through-
    # deinterleave loop runs inside the fused kernel (see kernels.py's
    # run_pipeline -- profiling measured the interpreted glue between
    # per-batch kernel calls at roughly two-thirds of this build's
    # steady state, and this is what deletes it). JIT warm-up above and
    # generation are excluded, same charter as every other build.
    steady_state_start = time.perf_counter()
    (batches_total, frame_count, batches_flagged, max_duty_cycle, max_mean_power,
     st_count, st_peak_sum, st_duration_sum, st_peak_min, st_peak_max,
     st_pri_sum, st_pri_count, track_count, next_track_id) = kernels.run_pipeline(
        i_all, q_all, idx_all, offsets,
        _SAMPLE_RATE_HZ, _NUM_BINS, bin_hz, threshold_sq,
        _POWER_THRESHOLD, _DUTY_CYCLE_THRESHOLD, _PRI_TOLERANCE_SECONDS,
        max_magnitude, sum_magnitude,
        track_id, track_pulse_count, track_last_start,
        track_pri_sum, track_pri_count, track_peak_sum,
    )
    steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    if batches_total < 0:
        print("[numba_monolith_app.py] ERROR: track capacity exceeded", file=sys.stderr)
        return 1
    batches = batches_total

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
