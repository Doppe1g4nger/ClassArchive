"""numba-jitted kernels: same algorithms as pulsecore/pulse_detector.py,
spectrogram.py, jammer.py, and iq_source.py, typed for @njit and
operating on flat numpy arrays instead of pulse_pb2 messages.

Why this preserves bit-for-bit correctness where numpy_variant/ can't:
numba compiles the *exact* Python source-level loop -- same operations,
same order, no reduction reordering -- down to a native loop. A
sequential accumulation like `re += ...` inside a `for` loop stays a
sequential accumulation after JIT compilation; it's compiled, not
rewritten. numpy's vectorized reductions (`array.sum()`, etc.) do the
opposite on purpose -- they reorder additions (pairwise summation, SIMD
lanes) specifically to go faster, which is *why* numpy is fast, but it
means results are only equal to the scalar version within floating-point
tolerance, not bit-identical. See numpy_variant/kernels.py's docstring
for that side of the comparison.

None of these functions know about pulse_pb2 at all. IQ generation
(generate_batch below) writes directly into numpy arrays instead of a
pulse_pb2.IQBatch -- there's no process/wire boundary in this
single-process build to justify routing through protobuf just to
immediately read the same values back out of it, so numba_monolith_app.py
doesn't. Only the detected pulse events (a much smaller array -- roughly
1,000/batch, not 10,000) get written into a pulse_pb2.PulseEventBatch,
since pulse_stats.py and deinterleaver.py are reused as-is and expect
that type.
"""
import numpy as np
from numba import njit

_UINT32_MAX = np.float64(0xFFFFFFFF)


_MASK32 = np.uint32(0xFFFFFFFF)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def spectrogram_bins(i_arr, q_arr, first_sample_index, sample_rate_hz, num_bins, bin_hz,
                      max_magnitude, sum_magnitude):
    """JIT-compiled port of spectrogram.py's process() loop -- same
    phasor-rotation recursion per bin, same sequential re/im
    accumulation, so it needs the same "read i/q once, not once per bin"
    fix spectrogram.py itself needed (see that module for why): done
    once by the caller, not here, since these arrays are already the
    per-batch extraction shared with detect_pulses/jammer_power above."""
    n = i_arr.shape[0]
    pi = 3.14159265358979323846

    for b in range(num_bins):
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


@njit(cache=True)
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
