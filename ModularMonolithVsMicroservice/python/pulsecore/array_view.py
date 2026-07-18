"""Protobuf-to-numpy conversion helper, used only by numpy_variant/ --
every other Python build in this repo (monolith, microservice,
multiproc, numba_variant) never imports this module and has no numpy
dependency at all. numba_variant doesn't need it because its IQ
generator writes directly into numpy arrays (see
numba_variant/kernels.py's generate_batch) instead of through
pulse_pb2.IQBatch in the first place -- see numpy_variant/kernels.py's
module docstring for why numpy_variant can't do the same thing and has
to pay this conversion instead.

Extracting a batch's `i`/`q`/`sample_index` fields into flat numpy
arrays is not free: `IQBatch.samples` is a repeated *message* field, not
a repeated scalar, so there is no bulk/zero-copy path in protobuf's
Python API for it -- this still costs one Python-level attribute read
per sample per array, the same class of cost the plain interpreted loop
pays. It is done here exactly once per batch and the resulting arrays
are reused across every stage that needs them (detector, spectrogram,
jammer all read `i`/`q`; only the detector also needs `sample_index`),
instead of every stage re-extracting its own copy.
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
