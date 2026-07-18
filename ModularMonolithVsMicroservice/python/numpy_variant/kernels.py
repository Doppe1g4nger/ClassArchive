"""numpy-vectorized kernels for the detector, spectrogram, and jammer --
the same three algorithms numba_variant/kernels.py JIT-compiles, rewritten
here as bulk array operations instead of compiled scalar loops. Two
honest consequences of that difference, both confirmed empirically (see
verify_numpy_variant.py), not just asserted:

1. **Not bit-identical to the rest of this repo.** `np.sum()` uses
   pairwise summation for arrays past a small size threshold, reordering
   the additions relative to the scalar builds' strict left-to-right
   accumulation; spectrogram_bins also evaluates each sample's absolute
   phase directly (`omega * n`) instead of the scalar/numba versions'
   recursive rotation, which is mathematically the same angle but
   numerically a different sequence of floating-point operations to get
   there. Floating-point addition and multiplication are not
   associative, so reordering them can change the last few bits of a
   result even when every input and the mathematical operation performed
   is identical. This is a well-understood property of vectorized
   numeric code, not a bug -- verified to stay within a relative error of
   less than 1e-9 against the scalar reference at this repo's scale, far
   below anything visible in the printed summaries, but real.
2. **The sqrt-avoidance trick from the scalar optimization pass is
   deliberately *not* used here.** Skipping `sqrt()` for below-threshold
   samples (see pulse_detector.cpp / pulse_detector.py) helps a scalar
   loop by cutting the number of sqrt *calls*. A vectorized sqrt call
   has no meaningful per-call overhead to cut -- it's one call either
   way -- and masking it to skip elements would force a conditional,
   non-vectorizable code path for no benefit. So `magnitude` below is
   computed unconditionally, for every sample, in one bulk `np.sqrt()`
   call: the optimization that helps the scalar/numba versions would
   actively hurt this one.

Like numba_variant/kernels.py, IQ generation is not vectorized here: the
xorshift32 RNG is a genuinely sequential recurrence (each state depends
on the previous one), which doesn't rewrite into bulk array ops without
a materially more advanced technique (e.g. jump-ahead via the RNG's
underlying linear-recurrence structure) that's out of proportion for
what this variant is demonstrating. numpy_monolith_app.py generates each
batch with the existing pulsecore.iq_source.SyntheticIQSource (into a
real pulse_pb2.IQBatch) and converts it once via
pulsecore.array_view.extract_iq -- a real cost numba_variant's
monolith app doesn't pay, since numba can JIT-compile that same
sequential generator loop directly with no rewrite required. That
asymmetry -- numba shrugs off a sequential bottleneck, numpy can't
without extra work -- is one of the more interesting differences between
the two approaches, not an oversight in this one.
"""
import numpy as np


def detect_pulses(i_arr, q_arr, idx_arr, threshold_sq, sample_rate_hz,
                   in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_sample_count):
    """Vectorized port of pulse_detector.py's process(). The threshold
    decision and magnitude computation are fully vectorized (the O(n)
    part, n=10,000/batch); finding pulse boundaries from the resulting
    boolean array and aggregating each one's peak/mean is a short loop
    over *events* (this repo's scale: ~1,000/batch), not samples --
    still a real reduction in Python-level iteration, just not a fully
    branch-free vectorization of the state machine itself, which would
    need to special-case a pulse straddling a batch boundary (rare in
    general, impossible at this repo's specific timing constants, but
    the code doesn't assume that -- see verify_numpy_variant.py's
    dedicated straddling test)."""
    n = i_arr.shape[0]
    mag_sq = i_arr * i_arr + q_arr * q_arr
    above = mag_sq >= threshold_sq
    magnitude = np.sqrt(mag_sq)  # see module docstring: unconditional on purpose

    above_i8 = above.astype(np.int8)
    transitions = np.empty(n, dtype=np.int8)
    transitions[0] = above_i8[0] - np.int8(1 if in_pulse else 0)
    if n > 1:
        np.subtract(above_i8[1:], above_i8[:-1], out=transitions[1:])

    rising = np.flatnonzero(transitions == 1)
    falling = np.flatnonzero(transitions == -1)

    ev_start = []
    ev_end = []
    ev_peak = []
    ev_mean = []
    ev_dur = []

    # A pulse already open when this batch started is closed by this
    # batch's first falling edge, if there is one -- its span runs from
    # sample 0 up to (not including) that edge, combined with whatever
    # peak/sum/count already accrued in a previous batch.
    carried_closed = False
    if in_pulse and falling.size > 0:
        f = int(falling[0])
        seg = magnitude[:f]
        seg_count = f
        total_count = pulse_sample_count + seg_count
        total_peak = max(pulse_peak, seg.max()) if seg_count > 0 else pulse_peak
        total_sum = pulse_sum + (float(seg.sum()) if seg_count > 0 else 0.0)
        ev_start.append(pulse_start)
        ev_end.append(int(idx_arr[f]))
        ev_peak.append(total_peak)
        ev_mean.append(total_sum / total_count)
        ev_dur.append(total_count / sample_rate_hz)
        carried_closed = True

    # Every remaining rising edge pairs 1:1, in order, with whichever
    # falling edges weren't consumed above -- both spans are fully
    # contained within this batch, so no carried state to combine.
    falling_for_new = falling[1:] if carried_closed else falling
    for r, f in zip(rising, falling_for_new):
        r = int(r)
        f = int(f)
        seg = magnitude[r:f]
        cnt = f - r
        ev_start.append(int(idx_arr[r]))
        ev_end.append(int(idx_arr[f]))
        ev_peak.append(float(seg.max()))
        ev_mean.append(float(seg.sum()) / cnt)
        ev_dur.append(cnt / sample_rate_hz)

    # Carry state into the next batch: still above threshold at the very
    # last sample means a pulse is open and unclosed at batch end.
    if bool(above[-1]):
        if rising.size > 0 and (falling.size == 0 or rising[-1] > falling[-1]):
            r = int(rising[-1])
            seg = magnitude[r:]
            new_in_pulse = True
            new_pulse_start = int(idx_arr[r])
            new_pulse_peak = float(seg.max())
            new_pulse_sum = float(seg.sum())
            new_pulse_sample_count = n - r
        else:
            # No rising edge at all this batch -- the entire batch
            # continues a pulse that was already open when it started.
            new_in_pulse = True
            new_pulse_start = pulse_start
            new_pulse_peak = max(pulse_peak, float(magnitude.max()))
            new_pulse_sum = pulse_sum + float(magnitude.sum())
            new_pulse_sample_count = pulse_sample_count + n
    else:
        new_in_pulse = False
        new_pulse_start = 0
        new_pulse_peak = 0.0
        new_pulse_sum = 0.0
        new_pulse_sample_count = 0

    return (
        np.array(ev_start, dtype=np.uint64),
        np.array(ev_end, dtype=np.uint64),
        np.array(ev_peak, dtype=np.float64),
        np.array(ev_mean, dtype=np.float64),
        np.array(ev_dur, dtype=np.float64),
        new_in_pulse,
        new_pulse_start,
        new_pulse_peak,
        new_pulse_sum,
        new_pulse_sample_count,
    )


def spectrogram_bins(i_arr, q_arr, first_sample_index, sample_rate_hz, num_bins, bin_hz,
                      max_magnitude, sum_magnitude):
    """Vectorized port of spectrogram.py's process(). Evaluates each
    sample's absolute phase directly (`omega * n`) instead of the
    scalar/numba versions' recursive per-sample rotation -- the same
    angle mathematically, a different (and here, vectorizable) sequence
    of floating-point operations to reach it. See module docstring for
    why that means this one isn't bit-identical to the others."""
    n = i_arr.shape[0]
    sample_offsets = np.arange(n, dtype=np.float64)

    for b in range(num_bins):
        freq_hz = (b + 0.5) * bin_hz
        omega = 2.0 * np.pi * freq_hz / sample_rate_hz
        phase = omega * (first_sample_index + sample_offsets)
        rot_re = np.cos(phase)
        rot_im = -np.sin(phase)

        re = float(np.sum(i_arr * rot_re - q_arr * rot_im))
        im = float(np.sum(i_arr * rot_im + q_arr * rot_re))

        magnitude = (re * re + im * im) ** 0.5 / n
        if magnitude > max_magnitude[b]:
            max_magnitude[b] = magnitude
        sum_magnitude[b] += magnitude

    return max_magnitude, sum_magnitude


def jammer_power(i_arr, q_arr, power_threshold):
    """Vectorized port of jammer.py's process()."""
    power = i_arr * i_arr + q_arr * q_arr
    power_sum = float(np.sum(power))
    over_threshold = int(np.count_nonzero(power >= power_threshold))
    return power_sum, over_threshold
