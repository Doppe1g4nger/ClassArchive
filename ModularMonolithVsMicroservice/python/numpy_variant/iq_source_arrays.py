"""Array-native port of pulsecore/iq_source.py for the numpy variant,
eliminating the pulse_pb2.IQBatch round-trip that cProfile measured as
this variant's single biggest cost: generating through protobuf and then
converting with array_view.extract_iq spent ~1.3s of the variant's
~2.1s profiled runtime on generation + extraction -- more than the
vectorized kernels themselves. There's no wire boundary inside this
single-process build, so nothing requires the samples to ever exist as
protobuf messages; this generator writes straight into numpy arrays,
the same choice numba_variant/kernels.py's generate_batch already made.

What's vectorized and what isn't, and why:

- sample_index is just arange(cursor, cursor+count) -- fully vectorized.
- The pulse/gap envelope is periodic in the sample index -- fully
  vectorized with a mask.
- The noise is NOT vectorized: xorshift32 is a sequential recurrence
  (each state depends on the previous one), the same reason
  numpy_variant doesn't vectorize generation at all elsewhere -- see
  kernels.py's module docstring. The Python loop below is only the RNG
  and two array stores per sample; everything that *can* leave the
  loop has.

Bit-identity with SyntheticIQSource: the same xorshift sequence in the
same order produces the same noise doubles; the envelope mask produces
the same component values (pulse_component or 0.0); and the final
`component + noise` is one IEEE double addition per element whether
numpy or the interpreter performs it. Verified against the protobuf
generator sample-for-sample by verify_numpy_variant.py, not just
argued.
"""
import numpy as np

# Same constants as pulsecore/iq_source.py, imported rather than copied
# so the two generators can't silently drift apart.
from pulsecore.iq_source import (
    _BATCH_SIZE,
    _GAP_SAMPLES,
    _PULSE_SAMPLES,
    _PULSE_COMPONENT,
    _NOISE_AMPLITUDE,
    _MASK32,
    _UINT32_MAX,
)


class SyntheticIQSourceArrays:
    def __init__(self, sample_rate_hz: float, num_pulses: int, seed: int = 42):
        self._sample_rate_hz = sample_rate_hz
        period = _GAP_SAMPLES + _PULSE_SAMPLES
        self._total_samples = num_pulses * period + _GAP_SAMPLES
        self._sample_cursor = 0
        self._rng_state = seed if seed != 0 else 1
        # Reused across batches; next_batch returns views of these.
        self._noise_i = np.empty(_BATCH_SIZE, dtype=np.float64)
        self._noise_q = np.empty(_BATCH_SIZE, dtype=np.float64)

    def next_batch(self):
        """Returns (i, q, sample_index) numpy arrays for the next batch,
        or None when every requested pulse has been emitted. The arrays
        are views into buffers reused by the next call -- callers (the
        numpy kernels) never hold them across batches."""
        if self._sample_cursor >= self._total_samples:
            return None

        cursor = self._sample_cursor
        count = min(_BATCH_SIZE, self._total_samples - cursor)
        period = _GAP_SAMPLES + _PULSE_SAMPLES

        # Sequential part: only the RNG recurrence and two stores per
        # sample live in the loop.
        noise_i = self._noise_i
        noise_q = self._noise_q
        rng = self._rng_state
        mask = _MASK32
        uint32_max = _UINT32_MAX
        amp2 = 2.0 * _NOISE_AMPLITUDE
        for k in range(count):
            rng = (rng ^ (rng << 13)) & mask
            rng = (rng ^ (rng >> 17)) & mask
            rng = (rng ^ (rng << 5)) & mask
            noise_i[k] = ((rng / uint32_max) - 0.5) * amp2
            rng = (rng ^ (rng << 13)) & mask
            rng = (rng ^ (rng >> 17)) & mask
            rng = (rng ^ (rng << 5)) & mask
            noise_q[k] = ((rng / uint32_max) - 0.5) * amp2
        self._rng_state = rng

        # Vectorized part: indices, envelope, and the final additions.
        idx = np.arange(cursor, cursor + count, dtype=np.uint64)
        in_pulse = (idx % np.uint64(period)) >= np.uint64(_GAP_SAMPLES)
        component = np.where(in_pulse, _PULSE_COMPONENT, 0.0)
        i_arr = component + noise_i[:count]
        q_arr = component + noise_q[:count]

        self._sample_cursor = cursor + count
        return i_arr, q_arr, idx
