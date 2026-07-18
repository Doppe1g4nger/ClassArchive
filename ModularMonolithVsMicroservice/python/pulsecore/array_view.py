"""Protobuf-to-numpy conversion helper. As of the second
profile-driven optimization pass, only numpy_variant's *verification
tool* (verify_numpy_variant.py) still uses this -- it needs to run the
scalar pulsecore reference and the vectorized kernels over the exact
same protobuf-generated batches to compare them. No benchmarked code
path imports it anymore: numpy_monolith_app.py now generates straight
into numpy arrays (see numpy_variant/iq_source_arrays.py) precisely
because cProfile measured this conversion, plus the protobuf generation
feeding it, as that variant's single biggest cost.

It's kept (rather than inlined into the verify script) as documentation
of *why* that cost exists: `IQBatch.samples` is a repeated *message*
field, not a repeated scalar, so there is no bulk/zero-copy path in
protobuf's Python API for it -- extraction costs one Python-level
attribute read per sample per array, the same class of cost the plain
interpreted loop pays.
"""
import numpy as np

from pulsecore import pulse_pb2


def extract_iq(batch: "pulse_pb2.IQBatch"):
    """Returns (i, q, sample_index) as float64/float64/uint64 numpy
    arrays, one element per sample in `batch`, in order."""
    samples = batch.samples
    n = len(samples)
    i = np.fromiter((s.i for s in samples), dtype=np.float64, count=n)
    q = np.fromiter((s.q for s in samples), dtype=np.float64, count=n)
    sample_index = np.fromiter(
        (s.sample_index for s in samples), dtype=np.uint64, count=n
    )
    return i, q, sample_index
