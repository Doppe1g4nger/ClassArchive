"""Small-bin magnitude spectrum estimator -- Python port of
common/src/spectrogram.cpp / common/include/spectrogram.h.

Includes the phasor-rotation optimization from that file's history: a
running complex value is rotated by one fixed multiply per sample instead
of calling cos()/sin() for every sample. Reintroducing the naive
cos()/sin()-per-sample version here would conflate "Python is slower at
calling library functions in a loop" with "Python is slower at plain
arithmetic in a loop" -- porting the already-optimized algorithm keeps the
comparison to the latter, which is the more meaningful one.
"""
import math

from pulsecore import pulse_pb2

_PI = 3.14159265358979323846


class SpectrogramAnalyzer:
    def __init__(self, sample_rate_hz: float, num_bins: int = 8):
        self._sample_rate_hz = sample_rate_hz
        self._num_bins = num_bins
        self._bin_hz = sample_rate_hz / (2.0 * num_bins)
        self._max_magnitude = [0.0] * num_bins
        self._sum_magnitude = [0.0] * num_bins
        self._frame_count = 0

    def process(self, batch: "pulse_pb2.IQBatch", out: "pulse_pb2.SpectrogramSummary") -> None:
        samples = batch.samples
        n = len(samples)
        if n > 0:
            first_sample_index = samples[0].sample_index

            # Read every sample out of the protobuf message exactly once,
            # into plain Python complex numbers, instead of once per bin
            # -- the loop below runs this batch's samples through all
            # num_bins correlators, and s.i/s.q are protobuf-generated
            # property accessors, not free attribute reads the way a C++
            # struct member is (a C++ compiler would hoist the redundant
            # reads automatically; CPython won't, so it's done by hand).
            #
            # complex, not an (i, q) pair, because the correlator's inner
            # loop *is* complex arithmetic: the four-multiply/two-add
            # update below is exactly (i + jq) * rot, and the phasor
            # advance is exactly rot * step. CPython evaluates a complex
            # product in C with the same component formulas the expanded
            # scalar code used -- (ac - bd) + j(ad + bc), same operations,
            # same order, verified bit-identical against the scalar
            # version, not just assumed -- so this halves the interpreted
            # bytecode per sample without changing a single output bit.
            # cProfile put this loop at 51% of the whole monolith's
            # runtime, which is what made it worth this treatment.
            samples_c = [complex(s.i, s.q) for s in samples]
            max_magnitude = self._max_magnitude
            sum_magnitude = self._sum_magnitude
            cos = math.cos
            sin = math.sin
            sqrt = math.sqrt

            for b in range(self._num_bins):
                freq_hz = (b + 0.5) * self._bin_hz
                omega = 2.0 * _PI * freq_hz / self._sample_rate_hz

                # Seed the phasor at this batch's first sample_index, then
                # advance it by one fixed complex multiply per sample
                # instead of recomputing cos()/sin() from scratch each
                # time -- see the module docstring.
                start_phase = omega * first_sample_index
                rot = complex(cos(start_phase), -sin(start_phase))
                step = complex(cos(omega), -sin(omega))

                acc = 0j
                for s in samples_c:
                    acc += s * rot
                    rot *= step

                # Not abs(acc): CPython's complex abs() goes through
                # hypot(), which rounds differently than the explicit
                # sqrt-of-sum-of-squares the scalar version used --
                # keeping the exact expression keeps the output
                # bit-identical.
                re = acc.real
                im = acc.imag
                magnitude = sqrt(re * re + im * im) / n
                if magnitude > max_magnitude[b]:
                    max_magnitude[b] = magnitude
                sum_magnitude[b] += magnitude
            self._frame_count += 1

        out.Clear()
        out.bin_hz = self._bin_hz
        out.frame_count = self._frame_count
        for b in range(self._num_bins):
            out.max_magnitude.append(self._max_magnitude[b])
            mean = self._sum_magnitude[b] / self._frame_count if self._frame_count > 0 else 0.0
            out.mean_magnitude.append(mean)
