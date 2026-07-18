"""Amplitude-threshold pulse detector -- Python port of
common/src/pulse_detector.cpp / common/include/pulse_detector.h.
"""
import math

from pulsecore import pulse_pb2


class PulseDetector:
    def __init__(self, amplitude_threshold: float, sample_rate_hz: float):
        self._threshold = amplitude_threshold
        self._sample_rate_hz = sample_rate_hz
        self._in_pulse = False
        self._pulse_start = 0
        self._pulse_peak = 0.0
        self._pulse_sum = 0.0
        self._pulse_sample_count = 0

    def process(self, batch: "pulse_pb2.IQBatch", out: "pulse_pb2.PulseEventBatch") -> None:
        for s in batch.samples:
            magnitude = math.sqrt(s.i * s.i + s.q * s.q)
            above = magnitude >= self._threshold

            if above and not self._in_pulse:
                self._in_pulse = True
                self._pulse_start = s.sample_index
                self._pulse_peak = magnitude
                self._pulse_sum = magnitude
                self._pulse_sample_count = 1
            elif above and self._in_pulse:
                self._pulse_peak = max(self._pulse_peak, magnitude)
                self._pulse_sum += magnitude
                self._pulse_sample_count += 1
            elif not above and self._in_pulse:
                event = out.events.add()
                event.start_sample = self._pulse_start
                event.end_sample = s.sample_index
                event.peak_amplitude = self._pulse_peak
                event.mean_amplitude = self._pulse_sum / self._pulse_sample_count
                event.duration_seconds = self._pulse_sample_count / self._sample_rate_hz
                self._in_pulse = False
