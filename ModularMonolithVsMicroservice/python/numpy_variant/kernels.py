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

Generation's history tracks this variant's whole optimization arc: it
first routed through pulse_pb2.IQBatch and converted with
array_view.extract_iq (measured as the variant's single biggest cost),
then wrote straight into numpy arrays with only the xorshift32
recurrence left as a Python loop (that loop then became the dominant
remaining cost), and finally vectorized the recurrence itself via GF(2)
jump-ahead -- the "materially more advanced technique" earlier versions
of this docstring deferred as out of proportion, adopted once profiling
showed it was the only thing left. numba never needed any of this: it
JIT-compiles the sequential loop as-is, an asymmetry between the two
approaches worth noticing. See iq_source_arrays.py for the jump-ahead
design and why the output is still bit-identical to pulsecore's
generator.
"""
import numpy as np


def detect_pulses(i_arr, q_arr, idx_arr, threshold_sq, sample_rate_hz,
                   in_pulse, pulse_start, pulse_peak, pulse_sum, pulse_sample_count):
    """Vectorized port of pulse_detector.py's process(). Fully
    vectorized as of the reduceat rework: the threshold decision, edge
    finding, AND the per-pulse peak/sum aggregation are all bulk array
    ops -- np.maximum.reduceat / np.add.reduceat over interleaved
    rising/falling boundaries reduce every wholly-in-batch pulse in two
    calls total, replacing what used to be a Python loop making two
    tiny-array reductions per event (~100k numpy calls per 50k-pulse
    run, measured as this kernel's dominant cost). Only the two
    carried-state edges (a pulse closed by this batch's first falling
    edge, a pulse left open at batch end) stay scalar -- at most one of
    each per batch.

    Exactness contract, stated precisely: start/end samples, durations,
    and peak amplitudes are exact for any input (maximum.reduceat
    reduces strictly sequentially; indices and counts are integers).
    Per-event mean_amplitude is exact for pulses up to 2 samples --
    which covers this repo's signal (2-sample pulses, see iq_source.h)
    and therefore keeps every cross-build output diff and the verify
    tool's exact event comparison green -- but np.add.reduceat
    reassociates sums for segments of 3+ samples (measured, not
    assumed: divergence from left-to-right summation starts at length
    3). The previous per-event-loop version had the same class of
    caveat at length 8+, via np.sum's SIMD lanes -- it was just
    undocumented. Nothing downstream ever reads mean_amplitude (stats
    uses peak/duration, the deinterleaver uses peak/start), so even for
    long-pulse inputs this can never reach any build's printed output.

    A pulse straddling a batch boundary is handled via the carried
    state (impossible at this repo's timing constants, supported
    anyway -- see verify_numpy_variant.py's dedicated test)."""
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

    # A pulse already open when this batch started is closed by this
    # batch's first falling edge, if there is one -- its span runs from
    # sample 0 up to (not including) that edge, combined with whatever
    # peak/sum/count already accrued in a previous batch.
    carried = None  # (start, end, peak, mean, duration) or None
    carried_closed = False
    if in_pulse and falling.size > 0:
        f = int(falling[0])
        seg = magnitude[:f]
        seg_count = f
        total_count = pulse_sample_count + seg_count
        total_peak = max(pulse_peak, seg.max()) if seg_count > 0 else pulse_peak
        total_sum = pulse_sum + (float(seg.sum()) if seg_count > 0 else 0.0)
        carried = (
            pulse_start,
            int(idx_arr[f]),
            total_peak,
            total_sum / total_count,
            total_count / sample_rate_hz,
        )
        carried_closed = True

    # Every remaining rising edge pairs 1:1, in order, with whichever
    # falling edges weren't consumed above (edges strictly alternate, so
    # rising[k] < falling_for_new[k] < rising[k+1]); each such pair is a
    # pulse wholly contained in this batch. Interleaving the two edge
    # arrays gives reduceat segment boundaries [r0:f0], [f0:r1], [r1:f1],
    # ... -- the pulses are every even-indexed segment.
    falling_for_new = falling[1:] if carried_closed else falling
    nev = falling_for_new.size
    if nev > 0:
        r = rising[:nev]
        f = falling_for_new
        bounds = np.empty(2 * nev, dtype=np.intp)
        bounds[0::2] = r
        bounds[1::2] = f
        counts = f - r
        new_start = idx_arr[r]
        new_end = idx_arr[f]
        new_peak = np.maximum.reduceat(magnitude, bounds)[0::2]
        new_mean = np.add.reduceat(magnitude, bounds)[0::2] / counts
        new_dur = counts / sample_rate_hz
    else:
        new_start = np.empty(0, dtype=np.uint64)
        new_end = np.empty(0, dtype=np.uint64)
        new_peak = np.empty(0, dtype=np.float64)
        new_mean = np.empty(0, dtype=np.float64)
        new_dur = np.empty(0, dtype=np.float64)

    if carried is not None:
        ev_start = np.concatenate(([carried[0]], new_start)).astype(np.uint64)
        ev_end = np.concatenate(([carried[1]], new_end)).astype(np.uint64)
        ev_peak = np.concatenate(([carried[2]], new_peak))
        ev_mean = np.concatenate(([carried[3]], new_mean))
        ev_dur = np.concatenate(([carried[4]], new_dur))
    else:
        ev_start = new_start.astype(np.uint64)
        ev_end = new_end.astype(np.uint64)
        ev_peak = new_peak
        ev_mean = new_mean
        ev_dur = new_dur

    # Carry state into the next batch: still above threshold at the very
    # last sample means a pulse is open and unclosed at batch end.
    if bool(above[-1]):
        if rising.size > 0 and (falling.size == 0 or rising[-1] > falling[-1]):
            r_last = int(rising[-1])
            seg = magnitude[r_last:]
            new_in_pulse = True
            new_pulse_start = int(idx_arr[r_last])
            new_pulse_peak = float(seg.max())
            new_pulse_sum = float(seg.sum())
            new_pulse_sample_count = n - r_last
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
        ev_start,
        ev_end,
        ev_peak,
        ev_mean,
        ev_dur,
        new_in_pulse,
        new_pulse_start,
        new_pulse_peak,
        new_pulse_sum,
        new_pulse_sample_count,
    )


# Per-(batch-length, geometry) cache of phasor tables: for each bin, the
# offset-dependent factor e^{-j*omega*k} for k in 0..n-1. The absolute
# phase omega*(first+k) factors as omega*first + omega*k, and only the
# first term changes between batches -- so the 2*num_bins*n
# transcendental evaluations that used to happen every batch (the bulk
# of this kernel's cost, per cProfile) happen once per batch *shape*
# instead, and each batch pays only 2*num_bins scalar cos/sin calls plus
# elementwise multiplies. In this repo there are exactly two shapes: the
# 10,000-sample full batch and the 8-sample tail.
_PHASE_TABLES = {}


def _phase_tables(n, sample_rate_hz, num_bins, bin_hz):
    key = (n, sample_rate_hz, num_bins, bin_hz)
    tables = _PHASE_TABLES.get(key)
    if tables is None:
        offsets = np.arange(n, dtype=np.float64)
        tables = []
        for b in range(num_bins):
            omega = 2.0 * np.pi * ((b + 0.5) * bin_hz) / sample_rate_hz
            tables.append((omega, np.cos(omega * offsets), -np.sin(omega * offsets)))
        _PHASE_TABLES[key] = tables
    return tables


def spectrogram_bins(i_arr, q_arr, first_sample_index, sample_rate_hz, num_bins, bin_hz,
                      max_magnitude, sum_magnitude):
    """Vectorized port of spectrogram.py's process(). Two evaluation-
    order departures from the scalar/numba versions, both inside this
    variant's documented not-bit-identical-but-tolerance-checked
    contract (see module docstring and verify_numpy_variant.py):

    1. Absolute phase instead of the recursive per-sample rotation
       (since this kernel's first version).
    2. The rotator e^{-j*omega*(first+k)} is built as the product of a
       per-batch scalar phasor e^{-j*omega*first} and a cached
       offset table e^{-j*omega*k} (see _phase_tables above), instead
       of a fresh cos/sin evaluation at every (bin, sample). Same angle
       by the trig addition identity; one more rounding step per
       element (a complex multiply), measured to keep the worst
       relative error vs. the scalar reference around 3e-11 -- well
       inside the 1e-9 verification gate."""
    n = i_arr.shape[0]
    tables = _phase_tables(n, sample_rate_hz, num_bins, bin_hz)

    for b in range(num_bins):
        omega, tab_re, tab_im = tables[b]
        phase0 = omega * first_sample_index
        r0_re = np.cos(phase0)
        r0_im = -np.sin(phase0)
        # (r0_re + j*r0_im) * (tab_re + j*tab_im), elementwise.
        rot_re = r0_re * tab_re - r0_im * tab_im
        rot_im = r0_re * tab_im + r0_im * tab_re

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
