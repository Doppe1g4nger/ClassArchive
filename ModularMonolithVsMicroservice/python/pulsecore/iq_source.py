"""Deterministic synthetic IQ generator -- Python port of
common/src/iq_source.cpp / common/include/iq_source.h. Uses the same
xorshift32 RNG and the same formulas, masking every intermediate value to
32 bits explicitly (Python ints don't wrap the way C++'s uint32_t does),
so it produces byte-for-byte the same sample stream as the C++ generator
for the same sample_rate_hz/num_pulses/seed.

Tuned for the same 1,000,000-pulse-per-second, 1000-microsecond-buffer
scale as the C++ side -- see that header's comment for the derivation.
Changing sample_rate_hz at a call site without changing the constants
below breaks the "exactly 1000 pulses/batch" property, same caveat as
the C++ version.
"""
import math

from pulsecore import pulse_pb2

_BATCH_SIZE = 10000
_GAP_SAMPLES = 8
_PULSE_SAMPLES = 2
_PULSE_AMPLITUDE = 10.0
_NOISE_AMPLITUDE = 0.5
_UINT32_MAX = 0xFFFFFFFF
_MASK32 = 0xFFFFFFFF
_SQRT2 = math.sqrt(2.0)
# I/Q split evenly across both components (see the in-pulse branch below);
# precomputed once instead of dividing by _SQRT2 on every sample, since
# there are only ever two possible values (this, or 0.0 in the gap).
_PULSE_COMPONENT = _PULSE_AMPLITUDE / _SQRT2


class SyntheticIQSource:
    def __init__(self, sample_rate_hz: float, num_pulses: int, seed: int = 42):
        self._sample_rate_hz = sample_rate_hz
        self._total_samples = num_pulses * (_GAP_SAMPLES + _PULSE_SAMPLES) + _GAP_SAMPLES
        self._sample_cursor = 0
        self._rng_state = seed if seed != 0 else 1

    def next_batch(self, batch: "pulse_pb2.IQBatch") -> bool:
        """Fills `batch` (typically frame.iq) with the next chunk of
        samples in place, mirroring NextBatch(pulse::IQBatch*) taking a
        pointer to caller-owned storage. Returns False once every
        requested pulse has been emitted."""
        if self._sample_cursor >= self._total_samples:
            return False

        batch.Clear()
        batch.sample_rate_hz = self._sample_rate_hz
        batch.first_sample_index = self._sample_cursor

        period = _GAP_SAMPLES + _PULSE_SAMPLES
        gap_samples = _GAP_SAMPLES
        pulse_component = _PULSE_COMPONENT
        noise_amplitude = _NOISE_AMPLITUDE
        mask = _MASK32
        uint32_max = _UINT32_MAX
        count = min(_BATCH_SIZE, self._total_samples - self._sample_cursor)
        cursor = self._sample_cursor
        rng = self._rng_state
        # phase cycles through [0, period) in lockstep with the sample
        # index; tracking it with an increment-and-wrap instead of
        # `idx % period` every iteration avoids a division per sample.
        phase = cursor % period

        # Packed columnar layout (see pulse.proto): build plain Python
        # lists in the loop, then hand each to protobuf in ONE bulk
        # extend() call -- upb copies a list of floats into a packed
        # array in C. The main branch's best effort was one add(**kwargs)
        # protobuf call per sample; this is two protobuf calls per batch.
        i_list = []
        q_list = []
        i_append = i_list.append
        q_append = q_list.append

        for _ in range(count):
            component = pulse_component if phase >= gap_samples else 0.0
            phase += 1
            if phase == period:
                phase = 0

            # Inlined xorshift32, bit-for-bit the C++ sequence -- see
            # the main branch's history for why the masking is explicit.
            rng = (rng ^ (rng << 13)) & mask
            rng = (rng ^ (rng >> 17)) & mask
            rng = (rng ^ (rng << 5)) & mask
            i_append(component + ((rng / uint32_max) - 0.5) * 2.0 * noise_amplitude)

            rng = (rng ^ (rng << 13)) & mask
            rng = (rng ^ (rng >> 17)) & mask
            rng = (rng ^ (rng << 5)) & mask
            q_append(component + ((rng / uint32_max) - 0.5) * 2.0 * noise_amplitude)

        batch.i.extend(i_list)
        batch.q.extend(q_list)

        self._sample_cursor = cursor + count
        self._rng_state = rng
        return True
