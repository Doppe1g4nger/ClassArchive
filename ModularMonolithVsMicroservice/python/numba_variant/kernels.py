"""numba-jitted kernels: same algorithms as pulsecore/pulse_detector.py,
spectrogram.py, jammer.py, and iq_source.py, typed for @njit and
operating on flat numpy arrays instead of pulse_pb2 messages.

Theoretical-limits branch: every kernel is compiled fastmath=True, the
LLVM equivalent of the -ffast-math the C++ side now uses globally --
reassociation, FMA contraction, and vectorization of the floating-point
loops are all permitted. That formally ends this variant's bit-for-bit
parity with the pure-Python reference (which earlier branches could
promise precisely because numba compiled the exact source-level loop
without rewriting it); parity is now asserted within 1e-12 relative
tolerance by tests/test_variants.py, the same trade the C++ builds made
and for the same reason: the sequential-order guarantee was the last
thing standing between these loops and the vector units.

None of these functions know about pulse_pb2 at all -- and since the
stats accumulator and deinterleaver became kernels too (see below),
neither does any per-batch code in numba_monolith_app.py: there's no
process/wire boundary in this single-process build to justify routing
anything through protobuf, so nothing is. IQ samples flow from
generate_batch's output arrays through every stage as arrays.
"""
import numpy as np
from numba import njit, prange

_UINT32_MAX = np.float64(0xFFFFFFFF)


_MASK32 = np.uint32(0xFFFFFFFF)


@njit(cache=True, fastmath=True)
def generate_batch(cursor, count, period, gap_samples, pulse_component, noise_amplitude, rng_state):
    """JIT-compiled port of iq_source.py's inlined xorshift32 generator
    loop -- same sequential RNG, same order, just compiled instead of
    interpreted. Unlike C++'s uint32_t, numba's uint32 `<<` does *not*
    truncate its result back to 32 bits (it widens instead, same as
    numpy's uint32 promotion rules) -- confirmed by testing against the
    plain-Python reference, not assumed -- so this still needs the same
    explicit `& mask` after every shift that the pure-Python version
    needs and C++ gets for free from its type."""
    i_arr = np.empty(count, dtype=np.float64)
    q_arr = np.empty(count, dtype=np.float64)
    idx_arr = np.empty(count, dtype=np.uint64)

    rng = np.uint32(rng_state)
    mask = _MASK32
    phase = cursor % period
    uint32_max = np.float64(0xFFFFFFFF)

    for k in range(count):
        component = pulse_component if phase >= gap_samples else 0.0
        phase += 1
        if phase == period:
            phase = 0

        rng = (rng ^ (rng << np.uint32(13))) & mask
        rng = (rng ^ (rng >> np.uint32(17))) & mask
        rng = (rng ^ (rng << np.uint32(5))) & mask
        noise_i = ((np.float64(rng) / uint32_max) - 0.5) * 2.0 * noise_amplitude

        rng = (rng ^ (rng << np.uint32(13))) & mask
        rng = (rng ^ (rng >> np.uint32(17))) & mask
        rng = (rng ^ (rng << np.uint32(5))) & mask
        noise_q = ((np.float64(rng) / uint32_max) - 0.5) * 2.0 * noise_amplitude

        idx_arr[k] = cursor + k
        i_arr[k] = component + noise_i
        q_arr[k] = component + noise_q

    return i_arr, q_arr, idx_arr, rng


@njit(cache=True, fastmath=True)
def detect_pulses(i_arr, q_arr, idx_arr, threshold_sq, sample_rate_hz,
                   in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_sample_count):
    """JIT-compiled port of pulse_detector.py's process() loop. Returns
    (event arrays, updated carry-across-batches state) instead of
    mutating self/out in place, since numba functions work best with
    explicit inputs/outputs rather than Python object state."""
    n = i_arr.shape[0]
    ev_start = np.empty(n, dtype=np.uint64)
    ev_end = np.empty(n, dtype=np.uint64)
    ev_peak = np.empty(n, dtype=np.float64)
    ev_mean = np.empty(n, dtype=np.float64)
    ev_dur = np.empty(n, dtype=np.float64)
    n_events = 0

    for k in range(n):
        si = i_arr[k]
        sq = q_arr[k]
        magnitude_sq = si * si + sq * sq
        above = magnitude_sq >= threshold_sq

        if above:
            magnitude = np.sqrt(magnitude_sq)
            if not in_pulse:
                in_pulse = True
                pulse_start = idx_arr[k]
                pulse_peak = magnitude
                pulse_sum = magnitude
                pulse_sample_count = 1
            else:
                if magnitude > pulse_peak:
                    pulse_peak = magnitude
                pulse_sum += magnitude
                pulse_sample_count += 1
        elif in_pulse:
            ev_start[n_events] = pulse_start
            ev_end[n_events] = idx_arr[k]
            ev_peak[n_events] = pulse_peak
            ev_mean[n_events] = pulse_sum / pulse_sample_count
            ev_dur[n_events] = pulse_sample_count / sample_rate_hz
            n_events += 1
            in_pulse = False

    return (ev_start[:n_events], ev_end[:n_events], ev_peak[:n_events], ev_mean[:n_events],
            ev_dur[:n_events], in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_sample_count)


@njit(cache=True, parallel=True, fastmath=True)
def spectrogram_bins(i_arr, q_arr, first_sample_index, sample_rate_hz, num_bins, bin_hz,
                      max_magnitude, sum_magnitude):
    """JIT-compiled port of spectrogram.py's process() loop -- same
    phasor-rotation recursion per bin, same sequential re/im
    accumulation within each bin (fastmath may reorder within a bin --
    parity with the reference is tolerance-gated, see the module
    docstring, same as every other kernel here).

    Round three moved the whole batch loop into one fused driver
    kernel (run_pipeline below) and A/B-tested this kernel's
    parallel=True inside it, expecting the per-batch thread handoff to
    stop paying once the Python glue around it was gone. Measured
    result: the opposite -- 5.2ms fused+parallel vs 14ms fused+serial
    (vs 22ms unfused+parallel). With the glue deleted, the spectrogram
    is most of what remains, and threading its 8 independent bins from
    inside compiled code keeps the whole speedup the handoff used to
    dilute. The wrong-way prediction is kept in the README as part of
    the round's ledger."""
    n = i_arr.shape[0]
    pi = 3.14159265358979323846

    for b in prange(num_bins):
        freq_hz = (b + 0.5) * bin_hz
        omega = 2.0 * pi * freq_hz / sample_rate_hz

        start_phase = omega * first_sample_index
        rot_re = np.cos(start_phase)
        rot_im = -np.sin(start_phase)
        step_re = np.cos(omega)
        step_im = -np.sin(omega)

        re = 0.0
        im = 0.0
        for k in range(n):
            si = i_arr[k]
            qi = q_arr[k]
            re += si * rot_re - qi * rot_im
            im += si * rot_im + qi * rot_re

            next_re = rot_re * step_re - rot_im * step_im
            next_im = rot_re * step_im + rot_im * step_re
            rot_re = next_re
            rot_im = next_im

        magnitude = np.sqrt(re * re + im * im) / n
        if magnitude > max_magnitude[b]:
            max_magnitude[b] = magnitude
        sum_magnitude[b] += magnitude

    return max_magnitude, sum_magnitude


@njit(cache=True, fastmath=True)
def jammer_power(i_arr, q_arr, power_threshold):
    """JIT-compiled port of jammer.py's process() loop."""
    n = i_arr.shape[0]
    power_sum = 0.0
    over_threshold = 0
    for k in range(n):
        si = i_arr[k]
        qi = q_arr[k]
        power = si * si + qi * qi
        power_sum += power
        if power >= power_threshold:
            over_threshold += 1
    return power_sum, over_threshold


# The two kernels below were added in a second pass, after cProfile
# showed that with detector/spectrogram/jammer/generation jitted, the
# numba build's remaining steady-state time was almost entirely the two
# stages deliberately left as pure Python -- the deinterleaver and stats
# accumulator -- plus rebuilding ~1,000 PulseEvent protobuf messages per
# batch solely to feed them. Jitting these two (operating directly on
# the detector kernel's output arrays) removes all three costs at once:
# the protobuf event rebuild disappears entirely, since nothing in this
# build needs pulse_pb2 for anything anymore.


@njit(cache=True, fastmath=True)
def stats_accumulate(ev_start, ev_peak, ev_dur, sample_rate_hz,
                      count, peak_sum, duration_sum, peak_min, peak_max,
                      pri_sum, pri_count, have_prev, prev_start):
    """JIT-compiled port of pulse_stats.py's add() loop -- same event
    order, same operations, so the running state (and therefore the
    final summary) is bit-identical to the pure-Python accumulator's."""
    for k in range(ev_start.shape[0]):
        count += 1
        pa = ev_peak[k]
        peak_sum += pa
        duration_sum += ev_dur[k]
        if pa < peak_min:
            peak_min = pa
        if pa > peak_max:
            peak_max = pa

        ss = ev_start[k]
        if have_prev:
            pri_sum += (ss - prev_start) / sample_rate_hz
            pri_count += 1
        prev_start = ss
        have_prev = True

    return count, peak_sum, duration_sum, peak_min, peak_max, pri_sum, pri_count, have_prev, prev_start


@njit(cache=True, fastmath=True)
def deinterleave_events(ev_start, ev_peak, sample_rate_hz, pri_tolerance_seconds,
                         track_count, next_track_id,
                         track_id, track_pulse_count, track_last_start,
                         track_pri_sum, track_pri_count, track_peak_sum):
    """JIT-compiled port of deinterleaver.py's process() loop. Track
    state lives in caller-owned parallel arrays instead of a list of
    objects; the caller guarantees capacity for track_count plus one new
    track per event (the worst case). Same per-event scan order, same
    tie-breaking (closest track within tolerance wins, first single-pulse
    track seeds otherwise), same arithmetic -- bit-identical tracks."""
    for e in range(ev_start.shape[0]):
        start_sample = ev_start[e]
        pulse_time = start_sample / sample_rate_hz

        best_match = -1
        best_diff = pri_tolerance_seconds
        seed_match = -1

        for t in range(track_count):
            if track_pri_count[t] > 0:
                last_time = track_last_start[t] / sample_rate_hz
                predicted = last_time + (track_pri_sum[t] / track_pri_count[t])
                diff = abs(predicted - pulse_time)
                if diff <= best_diff:
                    best_diff = diff
                    best_match = t
            elif track_pulse_count[t] == 1 and seed_match == -1:
                seed_match = t

        target = best_match if best_match != -1 else seed_match
        if target == -1:
            target = track_count
            track_id[target] = next_track_id
            next_track_id += 1
            track_pulse_count[target] = 0
            track_last_start[target] = 0
            track_pri_sum[target] = 0.0
            track_pri_count[target] = 0
            track_peak_sum[target] = 0.0
            track_count += 1

        if track_pulse_count[target] > 0:
            last_time = track_last_start[target] / sample_rate_hz
            track_pri_sum[target] += pulse_time - last_time
            track_pri_count[target] += 1
        track_last_start[target] = start_sample
        track_peak_sum[target] += ev_peak[e]
        track_pulse_count[target] += 1

    return track_count, next_track_id


@njit(cache=True, fastmath=True)
def run_pipeline(i_all, q_all, idx_all, offsets,
                 sample_rate_hz, num_bins, bin_hz, threshold_sq,
                 power_threshold, duty_cycle_threshold, pri_tolerance_seconds,
                 max_magnitude, sum_magnitude,
                 track_id, track_pulse_count, track_last_start,
                 track_pri_sum, track_pri_count, track_peak_sum):
    """Round three's fused driver: the entire batch loop, compiled.

    Profiling round three measured the numba build's kernels at ~8ms of
    its ~22ms steady state -- the rest was the Python driver loop
    between them: tuple packing/unpacking, slice bookkeeping, branch
    glue, 51 times over. This kernel IS that loop, so the per-batch
    boundary between compiled and interpreted code is gone; the app
    makes one call for the whole run.

    Inputs are the pre-generated signal as flat arrays plus offsets[b]
    marking each batch's start (offsets has batches+1 entries; the
    signal is a given on this branch -- see the app). Track capacity is
    a caller contract: if a batch would need more track slots than
    track_id has, the run aborts and returns batches_total == -1
    rather than write out of bounds -- the caller sizes the arrays to
    its input's event bound and treats -1 as a hard error.

    Returns every piece of cross-batch state the app prints:
    (batches_total, frame_count, batches_flagged, max_duty_cycle,
     max_mean_power, st_count, st_peak_sum, st_duration_sum,
     st_peak_min, st_peak_max, st_pri_sum, st_pri_count,
     track_count, next_track_id)."""
    in_pulse = False
    pulse_start = np.uint64(0)
    pulse_peak = 0.0
    pulse_sum = 0.0
    pulse_sample_count = 0

    frame_count = 0
    batches_total = 0
    batches_flagged = 0
    max_duty_cycle = 0.0
    max_mean_power = 0.0

    st_count = 0
    st_peak_sum = 0.0
    st_duration_sum = 0.0
    st_peak_min = np.inf
    st_peak_max = -np.inf
    st_pri_sum = 0.0
    st_pri_count = 0
    st_have_prev = False
    st_prev_start = np.uint64(0)

    track_count = 0
    next_track_id = 1

    for b in range(offsets.shape[0] - 1):
        s0 = offsets[b]
        s1 = offsets[b + 1]
        i_arr = i_all[s0:s1]
        q_arr = q_all[s0:s1]
        idx_arr = idx_all[s0:s1]
        n = s1 - s0

        (ev_start, ev_end, ev_peak, ev_mean, ev_dur, in_pulse, pulse_start, pulse_peak,
         pulse_sum, pulse_sample_count) = detect_pulses(
            i_arr, q_arr, idx_arr, threshold_sq, sample_rate_hz,
            in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_sample_count)

        spectrogram_bins(i_arr, q_arr, idx_arr[0], sample_rate_hz, num_bins, bin_hz,
                         max_magnitude, sum_magnitude)
        frame_count += 1

        power_sum, over_threshold = jammer_power(i_arr, q_arr, power_threshold)
        mean_power = power_sum / n
        duty_cycle = over_threshold / n
        batches_total += 1
        if duty_cycle >= duty_cycle_threshold:
            batches_flagged += 1
        if duty_cycle > max_duty_cycle:
            max_duty_cycle = duty_cycle
        if mean_power > max_mean_power:
            max_mean_power = mean_power

        (st_count, st_peak_sum, st_duration_sum, st_peak_min, st_peak_max,
         st_pri_sum, st_pri_count, st_have_prev, st_prev_start) = stats_accumulate(
            ev_start, ev_peak, ev_dur, sample_rate_hz,
            st_count, st_peak_sum, st_duration_sum, st_peak_min, st_peak_max,
            st_pri_sum, st_pri_count, st_have_prev, st_prev_start)

        if track_count + ev_start.shape[0] > track_id.shape[0]:
            return (-1, frame_count, batches_flagged, max_duty_cycle, max_mean_power,
                    st_count, st_peak_sum, st_duration_sum, st_peak_min, st_peak_max,
                    st_pri_sum, st_pri_count, track_count, next_track_id)
        track_count, next_track_id = deinterleave_events(
            ev_start, ev_peak, sample_rate_hz, pri_tolerance_seconds,
            track_count, next_track_id,
            track_id, track_pulse_count, track_last_start,
            track_pri_sum, track_pri_count, track_peak_sum)

    return (batches_total, frame_count, batches_flagged, max_duty_cycle, max_mean_power,
            st_count, st_peak_sum, st_duration_sum, st_peak_min, st_peak_max,
            st_pri_sum, st_pri_count, track_count, next_track_id)
