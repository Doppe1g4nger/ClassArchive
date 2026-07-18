"""numba_variant: a fourth Python architecture, answering "optimize
further with numba?" from the correctness-audit follow-up. JIT-compiles
the *exact same* scalar, per-sample algorithms already in
python/pulsecore/ (same loop structure, same order of operations) with
numba's @njit instead of interpreting them -- no rewrite to bulk array
ops, no reduction reordering, so kernels.py's output is bit-identical to
pulsecore's pure-Python output. See kernels.py's module docstring for
why that's specifically a numba property, not shared by numpy_variant/.

Only the three stages that touch the full 10,000-sample IQ batch
(detector, spectrogram, jammer) and IQ generation are JIT-compiled here.
pulse_stats and the deinterleaver operate on the much smaller `events`
list (roughly 1,000/batch, not 10,000) and are reused from pulsecore
as-is -- they were never the bottleneck this pass is targeting.
"""
