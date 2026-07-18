"""numpy_variant: a fifth Python architecture, alongside numba_variant/.
Same question -- "optimize further with numpy?" -- different answer:
instead of compiling the existing per-sample scalar loops (numba_variant's
approach), this rewrites the three stages that touch the full
10,000-sample IQ batch (detector, spectrogram, jammer) as bulk numpy
array operations. That's a genuine algorithm-shape change, not just a
faster execution of the same one, and it comes with a real, honest
consequence numba_variant doesn't have: **this variant's output is not
bit-identical to the other four Python builds**, only numerically
equivalent within a small floating-point tolerance. See
kernels.py's module docstring for exactly why, and
verify_numpy_variant.py for the actual measured tolerance.

pulse_stats and the deinterleaver are reused from pulsecore as-is: they
run over the much smaller `events` list (~1,000/batch, not 10,000
samples). With the RNG now vectorized too (GF(2) jump-ahead -- see
iq_source_arrays.py), cProfile puts this variant's remaining cost in
per-*event* work: the detector kernel's two tiny-array reductions per
detected pulse (~100k numpy calls per 50k-pulse run -- np.ufunc.reduceat
over the pulse boundaries would be the next fix, if one were wanted),
plus these two pure-Python stages and the protobuf event rebuild that
feeds them. (numba_variant crossed the equivalent threshold and jitted
its stats/deinterleaver; this variant's kernels are still slow enough
that doing the same here would buy proportionally less.)
"""
