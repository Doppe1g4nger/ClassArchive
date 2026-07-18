"""numba_variant: a fourth Python architecture, answering "optimize
further with numba?" from the correctness-audit follow-up. JIT-compiles
the *exact same* scalar, per-sample algorithms already in
python/pulsecore/ (same loop structure, same order of operations) with
numba's @njit instead of interpreting them -- no rewrite to bulk array
ops, no reduction reordering, so kernels.py's output is bit-identical to
pulsecore's pure-Python output. See kernels.py's module docstring for
why that's specifically a numba property, not shared by numpy_variant/.

All six pieces of per-batch work are JIT-compiled: IQ generation, the
three stages that touch the full 10,000-sample batch (detector,
spectrogram, jammer), and -- as of a second pass -- the stats
accumulator and deinterleaver too. Those last two operate on the much
smaller `events` stream (roughly 1,000/batch) and were originally
reused from pulsecore as pure Python on the theory that they'd never
matter; cProfile then showed that with everything else compiled, they
(plus rebuilding protobuf event messages just to feed them) had become
essentially *all* of this build's remaining steady-state time. Jitting
them removed that seam and the protobuf rebuild with it -- see
kernels.py's stats_accumulate/deinterleave_events. Output remains
byte-identical to every other build.
"""
