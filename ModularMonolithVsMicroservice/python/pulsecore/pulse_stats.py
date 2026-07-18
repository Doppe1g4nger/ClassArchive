"""Running summary-statistics accumulator -- Python port of
common/src/pulse_stats.cpp / common/include/pulse_stats.h.
"""
from pulsecore import pulse_pb2


class PulseStatsAccumulator:
    def __init__(self, sample_rate_hz: float):
        self._sample_rate_hz = sample_rate_hz
        self._count = 0
        self._peak_sum = 0.0
        self._duration_sum = 0.0
        self._peak_min = float("inf")
        self._peak_max = float("-inf")
        self._pri_sum = 0.0
        self._pri_count = 0
        self._have_prev_start = False
        self._prev_start_sample = 0

    def add(self, batch: "pulse_pb2.PulseEventBatch") -> None:
        for e in batch.events:
            self._count += 1
            self._peak_sum += e.peak_amplitude
            self._duration_sum += e.duration_seconds
            self._peak_min = min(self._peak_min, e.peak_amplitude)
            self._peak_max = max(self._peak_max, e.peak_amplitude)

            if self._have_prev_start:
                gap_samples = e.start_sample - self._prev_start_sample
                self._pri_sum += gap_samples / self._sample_rate_hz
                self._pri_count += 1
            self._prev_start_sample = e.start_sample
            self._have_prev_start = True

    def finalize(self) -> "pulse_pb2.PulseSummary":
        summary = pulse_pb2.PulseSummary()
        summary.pulse_count = self._count
        if self._count > 0:
            summary.mean_peak_amplitude = self._peak_sum / self._count
            summary.mean_duration_seconds = self._duration_sum / self._count
            summary.min_peak_amplitude = self._peak_min
            summary.max_peak_amplitude = self._peak_max
        if self._pri_count > 0:
            summary.mean_pri_seconds = self._pri_sum / self._pri_count
        return summary
