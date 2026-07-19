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
    accumulation within each bin. The *bins* run in parallel (prange):
    each bin's correlator is fully independent -- its own phasor, its
    own accumulators, its own output slots -- so threading them changes
    nothing about any bin's operation order (fastmath may, though --
    parity with the reference is tolerance-gated now, see the module
    docstring, same as every other kernel here). This was the numba build's
    largest remaining kernel by profile; parallel=True costs a
    per-call thread handoff, which the before/after numbers in the
    README weigh against the win."""
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
