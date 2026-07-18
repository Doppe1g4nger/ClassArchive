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


class SyntheticIQSource:
    def __init__(self, sample_rate_hz: float, num_pulses: int, seed: int = 42):
        self._sample_rate_hz = sample_rate_hz
        self._total_samples = num_pulses * (_GAP_SAMPLES + _PULSE_SAMPLES) + _GAP_SAMPLES
        self._sample_cursor = 0
        self._rng_state = seed if seed != 0 else 1

    def _next_noise(self) -> float:
        # xorshift32, matching common/src/iq_source.cpp bit-for-bit -- each
        # step explicitly masks to 32 bits since Python integers don't
        # wrap on their own the way C++'s uint32_t does.
        x = self._rng_state
        x = (x ^ (x << 13)) & _MASK32
        x = (x ^ (x >> 17)) & _MASK32
        x = (x ^ (x << 5)) & _MASK32
        self._rng_state = x
        unit = x / _UINT32_MAX
        return (unit - 0.5) * 2.0 * _NOISE_AMPLITUDE

    def next_batch(self, batch: "pulse_pb2.IQBatch") -> bool:
        """Fills `batch` (typically frame.iq) with the next chunk of
        samples in place, mirroring NextBatch(pulse::IQBatch*) taking a
        pointer to caller-owned storage. Returns False once every
        requested pulse has been emitted."""
        if self._sample_cursor >= self._total_samples:
            return False

        batch.Clear()
        batch.sample_rate_hz = self._sample_rate_hz

        period = _GAP_SAMPLES + _PULSE_SAMPLES
        count = min(_BATCH_SIZE, self._total_samples - self._sample_cursor)
        cursor = self._sample_cursor

        for i in range(count):
            idx = cursor + i
            phase = idx % period
            in_pulse = phase >= _GAP_SAMPLES
            amplitude = _PULSE_AMPLITUDE if in_pulse else 0.0
            component = amplitude / _SQRT2

            s = batch.samples.add()
            s.sample_index = idx
            s.i = component + self._next_noise()
            s.q = component + self._next_noise()

        self._sample_cursor = cursor + count
        return True
