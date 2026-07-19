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
samples). After four rounds of measured fixes (protobuf round-trip
removed, GF(2) jump-ahead RNG, reduceat event aggregation, cached
phase tables -- see kernels.py and iq_source_arrays.py), those two
pure-Python stages plus the protobuf event rebuild that feeds them ARE
this variant's remaining floor. Fixing that would mean either jitting
them (numba_variant's answer -- but then this variant stops being "what
numpy alone can do") or duplicating their algorithms as app-local array
code; both were judged worse than stating the floor honestly.
"""
