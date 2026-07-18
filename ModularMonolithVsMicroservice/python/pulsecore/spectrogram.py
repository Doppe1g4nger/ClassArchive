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
            for b in range(self._num_bins):
                freq_hz = (b + 0.5) * self._bin_hz
                omega = 2.0 * _PI * freq_hz / self._sample_rate_hz

                # Seed the phasor at this batch's first sample_index, then
                # advance it by one fixed complex multiply per sample
                # instead of recomputing cos()/sin() from scratch each
                # time -- see the module docstring.
                start_phase = omega * first_sample_index
                rot_re = math.cos(start_phase)
                rot_im = -math.sin(start_phase)
                step_re = math.cos(omega)
                step_im = -math.sin(omega)

                re = 0.0
                im = 0.0
                for s in samples:
                    re += s.i * rot_re - s.q * rot_im
                    im += s.i * rot_im + s.q * rot_re

                    next_re = rot_re * step_re - rot_im * step_im
                    next_im = rot_re * step_im + rot_im * step_re
                    rot_re = next_re
                    rot_im = next_im

                magnitude = math.sqrt(re * re + im * im) / n
                if magnitude > self._max_magnitude[b]:
                    self._max_magnitude[b] = magnitude
                self._sum_magnitude[b] += magnitude
            self._frame_count += 1

        out.Clear()
        out.bin_hz = self._bin_hz
        out.frame_count = self._frame_count
        for b in range(self._num_bins):
            out.max_magnitude.append(self._max_magnitude[b])
            mean = self._sum_magnitude[b] / self._frame_count if self._frame_count > 0 else 0.0
            out.mean_magnitude.append(mean)
