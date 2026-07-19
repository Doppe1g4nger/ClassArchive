"""Protobuf-to-numpy conversion helper. As of the second
profile-driven optimization pass, only numpy_variant's *verification
tool* (verify_numpy_variant.py) still uses this -- it needs to run the
scalar pulsecore reference and the vectorized kernels over the exact
same protobuf-generated batches to compare them. No benchmarked code
path imports it anymore: numpy_monolith_app.py now generates straight
into numpy arrays (see numpy_variant/iq_source_arrays.py) precisely
because cProfile measured this conversion, plus the protobuf generation
feeding it, as that variant's single biggest cost.

(On the main branch this function is also documentation of why the
conversion was expensive: there, `IQBatch.samples` is a repeated
*message* field with no bulk path, costing one Python-level attribute
read per sample per array. This branch's packed columnar schema is the
fix -- see extract_iq's docstring.)
"""
import numpy as np

from pulsecore import pulse_pb2


def extract_iq(batch: "pulse_pb2.IQBatch"):
    """Returns (i, q, sample_index) as float64/float64/uint64 numpy
    arrays, one element per sample in `batch`, in order. On the packed
    columnar schema (max-optimization branch) this is nearly free --
    np.asarray over a packed repeated double is a single bulk copy, and
    the indices are just an arange from first_sample_index -- which is
    itself a measure of what the main branch's one-message-per-sample
    layout was costing this conversion."""
    n = len(batch.i)
    i = np.asarray(batch.i, dtype=np.float64)
    q = np.asarray(batch.q, dtype=np.float64)
    first = batch.first_sample_index
    sample_index = np.arange(first, first + n, dtype=np.uint64)
    return i, q, sample_index
