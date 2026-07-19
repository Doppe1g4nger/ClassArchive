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
        # Same reasoning as PulseDetector.process(): hoist the running
        # state to locals for the loop, write back once at the end.
        count = self._count
        peak_sum = self._peak_sum
        duration_sum = self._duration_sum
        peak_min = self._peak_min
        peak_max = self._peak_max
        pri_sum = self._pri_sum
        pri_count = self._pri_count
        have_prev_start = self._have_prev_start
        prev_start_sample = self._prev_start_sample
        sample_rate_hz = self._sample_rate_hz

        # Columnar events (see pulse.proto): parallel arrays, bulk-copied
        # out of protobuf once, then plain-list iteration.
        starts = list(batch.start_sample)
        peaks = list(batch.peak_amplitude)
        durations = list(batch.duration_seconds)
        for k in range(len(starts)):
            count += 1
            peak_amplitude = peaks[k]
            peak_sum += peak_amplitude
            duration_sum += durations[k]
            if peak_amplitude < peak_min:
                peak_min = peak_amplitude
            if peak_amplitude > peak_max:
                peak_max = peak_amplitude

            start_sample = starts[k]
            if have_prev_start:
                pri_sum += (start_sample - prev_start_sample) / sample_rate_hz
                pri_count += 1
            prev_start_sample = start_sample
            have_prev_start = True

        self._count = count
        self._peak_sum = peak_sum
        self._duration_sum = duration_sum
        self._peak_min = peak_min
        self._peak_max = peak_max
        self._pri_sum = pri_sum
        self._pri_count = pri_count
        self._have_prev_start = have_prev_start
        self._prev_start_sample = prev_start_sample

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
