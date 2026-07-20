"""Small-bin magnitude spectrum estimator -- Python port of
common/src/spectrogram.cpp / common/include/spectrogram.h.

Theoretical-limits branch, round four: this port now uses the same
cached phase tables its C++ and numpy counterparts adopted rounds ago
-- and in CPython the payoff is structural, not arithmetic. With the
per-batch phasor factored OUT of the sum (sum(s_k * r0 * t_k) ==
r0 * sum(s_k * t_k)), each bin's whole correlator becomes

    sum(map(mul, samples, table), 0j) * r0

which runs entirely inside the interpreter's C internals: map()
iterates at C speed, complex.__mul__ multiplies at C speed, sum()
accumulates at C speed. The earlier phasor-recursion version -- kept
faithful to the C++ history and bit-identical to it -- executed two
interpreted bytecode statements per sample per bin; this version
executes zero. That bit-identical claim is the price: the table
entries are computed fresh from cos/sin instead of accumulated by
repeated complex multiply (less rounding drift, in fact), and the
factored r0 rounds once at the end, so parity with the C++ analyzer
is tolerance-level now -- which is this branch's contract everywhere
(printed outputs are unchanged at their 3-decimal precision, and the
variant parity tests gate the kernels against this reference at 1e-9).

The summation order is unchanged (sequential over samples), so this is
a rounding-level change, not a reordering.
"""
import math
from operator import mul

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
        # Per-batch-length cache of per-bin offset phasor tables
        # e^{-j*omega*k}, exactly like the C++ analyzer's table_n_
        # members -- this repo's signal has two batch lengths, so the
        # trig runs twice per process lifetime.
        self._tables: dict[int, list[list[complex]]] = {}

    def process(self, batch: "pulse_pb2.IQBatch", out: "pulse_pb2.SpectrogramSummary") -> None:
        n = len(batch.i)
        if n > 0:
            first_sample_index = batch.first_sample_index

            # One bulk read of the packed arrays into plain complex
            # numbers (protobuf accessors are not free attribute reads;
            # see this file's history of the same hoist).
            samples_c = [complex(si, qi) for si, qi in zip(batch.i, batch.q)]

            tables = self._tables.get(n)
            if tables is None:
                tables = []
                for b in range(self._num_bins):
                    omega = 2.0 * _PI * ((b + 0.5) * self._bin_hz) / self._sample_rate_hz
                    tables.append(
                        [complex(math.cos(omega * k), -math.sin(omega * k)) for k in range(n)]
                    )
                self._tables[n] = tables

            max_magnitude = self._max_magnitude
            sum_magnitude = self._sum_magnitude
            cos = math.cos
            sin = math.sin
            sqrt = math.sqrt

            for b in range(self._num_bins):
                omega = 2.0 * _PI * ((b + 0.5) * self._bin_hz) / self._sample_rate_hz
                start_phase = omega * first_sample_index
                r0 = complex(cos(start_phase), -sin(start_phase))

                # The entire correlator, at C speed -- see the module
                # docstring. Sequential accumulation, same order as the
                # explicit loop it replaced.
                acc = sum(map(mul, samples_c, tables[b]), 0j) * r0

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
