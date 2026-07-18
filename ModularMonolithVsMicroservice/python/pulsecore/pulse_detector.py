"""Amplitude-threshold pulse detector -- Python port of
common/src/pulse_detector.cpp / common/include/pulse_detector.h.
"""
import math

from pulsecore import pulse_pb2


class PulseDetector:
    def __init__(self, amplitude_threshold: float, sample_rate_hz: float):
        # Squared once here so process() can compare i*i+q*q against it
        # directly instead of computing sqrt(i*i+q*q) for every sample --
        # see pulse_detector.cpp for why that's exact, not approximate.
        self._threshold_sq = amplitude_threshold * amplitude_threshold
        self._sample_rate_hz = sample_rate_hz
        self._in_pulse = False
        self._pulse_start = 0
        self._pulse_peak = 0.0
        self._pulse_sum = 0.0
        self._pulse_sample_count = 0

    def process(self, batch: "pulse_pb2.IQBatch", out: "pulse_pb2.PulseEventBatch") -> None:
        # Hoisted to locals for the loop below: CPython resolves a local
        # variable (LOAD_FAST) faster than an attribute lookup on self
        # (LOAD_ATTR), and this loop runs once per IQ sample (10,000 per
        # batch). Written back to self once at the end, matching the
        # state this detector carries across batches (a pulse can
        # straddle a batch boundary).
        threshold_sq = self._threshold_sq
        sample_rate_hz = self._sample_rate_hz
        in_pulse = self._in_pulse
        pulse_start = self._pulse_start
        pulse_peak = self._pulse_peak
        pulse_sum = self._pulse_sum
        pulse_sample_count = self._pulse_sample_count
        add_event = out.events.add
        sqrt = math.sqrt

        for s in batch.samples:
            si = s.i
            sq = s.q
            magnitude_sq = si * si + sq * sq
            above = magnitude_sq >= threshold_sq

            # sqrt() only computed for samples that actually cross the
            # threshold -- this repo's synthetic signal keeps that to
            # ~20% of samples (its duty cycle, see iq_source.py).
            if above:
                magnitude = sqrt(magnitude_sq)
                if not in_pulse:
                    in_pulse = True
                    pulse_start = s.sample_index
                    pulse_peak = magnitude
                    pulse_sum = magnitude
                    pulse_sample_count = 1
                else:
                    if magnitude > pulse_peak:
                        pulse_peak = magnitude
                    pulse_sum += magnitude
                    pulse_sample_count += 1
            elif in_pulse:
                event = add_event()
                event.start_sample = pulse_start
                event.end_sample = s.sample_index
                event.peak_amplitude = pulse_peak
                event.mean_amplitude = pulse_sum / pulse_sample_count
                event.duration_seconds = pulse_sample_count / sample_rate_hz
                in_pulse = False

        self._in_pulse = in_pulse
        self._pulse_start = pulse_start
        self._pulse_peak = pulse_peak
        self._pulse_sum = pulse_sum
        self._pulse_sample_count = pulse_sample_count
