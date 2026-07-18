"""Array-native IQ generator for the numpy variant -- now fully
vectorized, including the RNG, via GF(2) jump-ahead.

This module has been through two designs, and the docstring history
matters for reading the benchmarks:

1. First it existed to eliminate the pulse_pb2 round-trip (generate
   through protobuf, convert with array_view.extract_iq) that cProfile
   measured as the variant's single biggest cost. That version
   vectorized the sample indices, the pulse/gap envelope, and the final
   additions, but kept the xorshift32 noise recurrence as a plain
   Python loop -- each state depends on the previous one, so it "doesn't
   rewrite into bulk array ops" without more machinery. Re-profiling
   then showed exactly what that reasoning predicted: the loop became
   the variant's dominant remaining cost (~330ms of ~565ms).

2. This version brings the machinery. xorshift32's step is *linear over
   GF(2)*: each of `s ^= s<<13; s ^= s>>17; s ^= s<<5` is an XOR of bit
   shifts, so one step is a 32x32 bit-matrix multiply, state(n+k) =
   M^k * state(n), and M^k is computable once by square-and-multiply.
   That turns "inherently sequential" into "sequential only if you
   evaluate it that way":

   - The 20,000 draws of a full batch are split into K=2,500 lanes of
     T=8 consecutive draws each. Lane j owns draws [j*T+1 .. (j+1)*T] --
     a *contiguous* chunk, so a row-major (K, T) grid flattens into
     exactly the scalar sequence order.
   - Within a batch, all K lanes advance together: one plain
     three-shift xorshift step applied elementwise to a uint32 array
     (numpy uint32 shifts truncate natively -- no masking needed), T
     times. ~150 numpy calls replace 20,000 interpreted iterations.
   - Between batches, every lane jumps forward by the same fixed
     distance (20,000 - T), one precomputed M^(20000-T) applied
     vectorized (32 masked XORs).
   - Lane seeds come from one scalar pre-run at construction time
     (recording every T-th state), excluded from the steady-state
     timer by the same rule that excludes numba's JIT warm-up and the
     monolith's dlopen(): a one-time setup cost, not per-batch work.
   - The final partial batch (8 samples at this repo's constants) falls
     back to the plain scalar loop, continuing from lane 0's state --
     which, by the lane layout, *is* the sequence state at the start of
     the next batch.

Everything here is exact integer XOR/shift arithmetic -- no floating
point until the final draws-to-noise conversion, which is elementwise
and identical to the scalar version's. The output is therefore
bit-identical to pulsecore's generator: the same sequence, evaluated in
a different order. That's enforced, not assumed -- by
verify_numpy_variant.py's generator check and the unit tests in
python/tests/, which compare against the protobuf generator sample for
sample and don't care how the numbers were produced.
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

# Lane geometry: K lanes x T draws = one full batch's 2*_BATCH_SIZE
# draws (i and q per sample). T trades numpy call count (smaller T =
# fewer step calls) against lane-seeding cost at construction (fixed:
# one scalar pre-run of K*T steps either way); 8 keeps the per-batch
# call count near its floor without exotic tuning.
_DRAWS_PER_BATCH = 2 * _BATCH_SIZE
_T = 8
_K = _DRAWS_PER_BATCH // _T
assert _K * _T == _DRAWS_PER_BATCH


def _scalar_step(s):
    s = (s ^ (s << 13)) & _MASK32
    s = (s ^ (s >> 17)) & _MASK32
    s = (s ^ (s << 5)) & _MASK32
    return s


def _step_matrix():
    """xorshift32's one-step transition as a GF(2) matrix, represented
    by its 32 basis-vector images (column b = step applied to 1<<b).
    Valid because every operation in the step is an XOR of shifts --
    linear over GF(2), so the step is fully determined by where it
    sends each basis vector."""
    return [_scalar_step(1 << b) for b in range(32)]


def _mat_apply(cols, s):
    out = 0
    b = 0
    while s:
        if s & 1:
            out ^= cols[b]
        s >>= 1
        b += 1
    return out


def _mat_mul(a_cols, b_cols):
    return [_mat_apply(a_cols, c) for c in b_cols]


def _mat_pow(cols, n):
    """M^n by square-and-multiply -- ~log2(n) 32x32 GF(2) multiplies,
    done once at import time, in plain Python integers."""
    result = [1 << b for b in range(32)]  # identity
    base = cols
    while n:
        if n & 1:
            result = _mat_mul(base, result)
        base = _mat_mul(base, base)
        n >>= 1
    return result


# Precomputed once at import: the between-batch lane jump. After a lane
# emits its T draws, its next chunk starts (20,000 - T) states later --
# the same distance for every lane, which is what makes one shared
# matrix (applied elementwise) sufficient.
_JUMP_COLS = np.array(_mat_pow(_step_matrix(), _DRAWS_PER_BATCH - _T), dtype=np.uint32)


def _jump_lanes(lanes):
    """Applies the precomputed jump matrix to every lane at once: for
    each of the 32 state bits, XOR in that bit's column image wherever
    the bit is set. 32 masked XORs regardless of lane count."""
    out = np.zeros_like(lanes)
    for b in range(32):
        bit = (lanes >> np.uint32(b)) & np.uint32(1)
        out ^= bit * _JUMP_COLS[b]
    return out


class SyntheticIQSourceArrays:
    def __init__(self, sample_rate_hz: float, num_pulses: int, seed: int = 42):
        self._sample_rate_hz = sample_rate_hz
        period = _GAP_SAMPLES + _PULSE_SAMPLES
        self._total_samples = num_pulses * period + _GAP_SAMPLES
        self._sample_cursor = 0

        # One scalar pre-run seeds the K lanes with the states at
        # positions 0, T, 2T, ..., (K-1)*T -- construction-time cost
        # (~20k plain-Python steps), outside the steady-state timer.
        s = seed if seed != 0 else 1
        lanes = np.empty(_K, dtype=np.uint32)
        lanes[0] = s
        for j in range(1, _K):
            for _ in range(_T):
                s = _scalar_step(s)
            lanes[j] = s
        self._lanes = lanes
        # Reused output grid for the vectorized path.
        self._grid = np.empty((_K, _T), dtype=np.uint32)

    def next_batch(self):
        """Returns (i, q, sample_index) numpy arrays for the next batch,
        or None when every requested pulse has been emitted. The arrays
        are views into buffers reused by the next call -- callers (the
        numpy kernels) never hold them across batches."""
        if self._sample_cursor >= self._total_samples:
            return None

        cursor = self._sample_cursor
        count = min(_BATCH_SIZE, self._total_samples - cursor)

        if count == _BATCH_SIZE:
            draws = self._full_batch_draws()
        else:
            draws = self._tail_draws(count)

        # Draws alternate i-noise, q-noise -- even/odd strided views of
        # the flattened sequence, then the same elementwise conversion
        # the scalar generator applies per draw.
        floats = ((draws / np.float64(_UINT32_MAX)) - 0.5) * (2.0 * _NOISE_AMPLITUDE)
        noise_i = floats[0::2]
        noise_q = floats[1::2]

        # Vectorized envelope: indices and the pulse/gap mask.
        period = _GAP_SAMPLES + _PULSE_SAMPLES
        idx = np.arange(cursor, cursor + count, dtype=np.uint64)
        in_pulse = (idx % np.uint64(period)) >= np.uint64(_GAP_SAMPLES)
        component = np.where(in_pulse, _PULSE_COMPONENT, 0.0)
        i_arr = component + noise_i
        q_arr = component + noise_q

        self._sample_cursor = cursor + count
        return i_arr, q_arr, idx

    def _full_batch_draws(self):
        """All 20,000 draws of a full batch: T vectorized xorshift steps
        across K lanes, then one vectorized jump to reposition the lanes
        for the next batch."""
        grid = self._grid
        col = self._lanes
        for t in range(_T):
            col = col ^ (col << np.uint32(13))
            col = col ^ (col >> np.uint32(17))
            col = col ^ (col << np.uint32(5))
            grid[:, t] = col
        self._lanes = _jump_lanes(col)
        return grid.reshape(-1)

    def _tail_draws(self, count):
        """The final partial batch, via the plain scalar loop. Lane 0's
        state is, by the lane layout, exactly the sequence state at this
        point -- each batch's lane 0 owns the first chunk, and the jump
        repositions it to the start of the next batch's draws."""
        n_draws = 2 * count
        out = np.empty(n_draws, dtype=np.uint32)
        s = int(self._lanes[0])
        for k in range(n_draws):
            s = _scalar_step(s)
            out[k] = s
        # No lane update needed: a tail batch is only ever the last one
        # (count < _BATCH_SIZE implies the signal is exhausted).
        return out
