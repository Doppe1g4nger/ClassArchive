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

        period = _GAP_SAMPLES + _PULSE_SAMPLES
        gap_samples = _GAP_SAMPLES
        pulse_component = _PULSE_COMPONENT
        noise_amplitude = _NOISE_AMPLITUDE
        mask = _MASK32
        uint32_max = _UINT32_MAX
        count = min(_BATCH_SIZE, self._total_samples - self._sample_cursor)
        cursor = self._sample_cursor
        add_sample = batch.samples.add
        rng = self._rng_state
        # phase cycles through [0, period) in lockstep with the sample
        # index; tracking it with an increment-and-wrap instead of
        # `idx % period` every iteration avoids a division per sample.
        phase = cursor % period

        for i in range(count):
            component = pulse_component if phase >= gap_samples else 0.0
            phase += 1
            if phase == period:
                phase = 0

            # Inlined xorshift32 (was a self._next_noise() method called
            # twice per sample -- 20,000 bound-method calls per
            # 10,000-sample batch just for RNG dispatch). A C++ compiler
            # inlines the equivalent private-method calls automatically at
            # -O3 (see iq_source.cpp); CPython never inlines method calls,
            # so this does it by hand to get the same effect. Still
            # matches common/src/iq_source.cpp bit-for-bit: each step
            # explicitly masks to 32 bits since Python integers don't wrap
            # on their own the way C++'s uint32_t does, and i's noise is
            # drawn before q's, same order as the two original calls.
            rng = (rng ^ (rng << 13)) & mask
            rng = (rng ^ (rng >> 17)) & mask
            rng = (rng ^ (rng << 5)) & mask
            noise_i = ((rng / uint32_max) - 0.5) * 2.0 * noise_amplitude

            rng = (rng ^ (rng << 13)) & mask
            rng = (rng ^ (rng >> 17)) & mask
            rng = (rng ^ (rng << 5)) & mask
            noise_q = ((rng / uint32_max) - 0.5) * 2.0 * noise_amplitude

            # One add() call with field kwargs instead of add() plus
            # three attribute assignments -- upb constructs and fills
            # the sample in a single C call, saving three descriptor
            # lookups and three setattr dispatches per sample (30,000
            # interpreted operations per batch; cProfile showed this
            # loop's protobuf traffic among the monolith's top line
            # items). Field values are identical either way.
            add_sample(
                sample_index=cursor + i,
                i=component + noise_i,
                q=component + noise_q,
            )

        self._sample_cursor = cursor + count
        self._rng_state = rng
        return True
