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
samples) and have never shown up as this variant's bottleneck --
cProfile puts its remaining cost in the sequential RNG loop, not these
stages. (numba_variant started from the same reasoning and later jitted
its stats/deinterleaver anyway, because *there* the compiled kernels got
fast enough that the pure-Python seams became the dominant remaining
cost -- a threshold this variant's slower kernels never crossed.)
"""
