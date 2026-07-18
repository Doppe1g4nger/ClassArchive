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

            # Read every sample's i/q out of the protobuf message exactly
            # once, into plain Python floats, instead of once per bin --
            # the loop below runs this batch's samples through all
            # num_bins correlators, and s.i/s.q are protobuf-generated
            # property accessors, not free attribute reads the way a C++
            # struct member is. Re-reading them from the message on every
            # (bin, sample) pair means num_bins times the attribute-access
            # cost for no benefit, since the values never change across
            # bins; a C++ compiler would hoist this automatically for an
            # inlined getter, CPython won't, so it's done by hand here.
            i_vals = [s.i for s in samples]
            q_vals = [s.q for s in samples]
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
                rot_re = cos(start_phase)
                rot_im = -sin(start_phase)
                step_re = cos(omega)
                step_im = -sin(omega)

                re = 0.0
                im = 0.0
                for si, qi in zip(i_vals, q_vals):
                    re += si * rot_re - qi * rot_im
                    im += si * rot_im + qi * rot_re

                    next_re = rot_re * step_re - rot_im * step_im
                    next_im = rot_re * step_im + rot_im * step_re
                    rot_re = next_re
                    rot_im = next_im

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
