"""PRI-based emitter track grouping -- Python port of
common/src/deinterleaver.cpp / common/include/deinterleaver.h.
"""
from pulsecore import pulse_pb2


class _Track:
    __slots__ = (
        "track_id",
        "pulse_count",
        "last_start_sample",
        "pri_sum_seconds",
        "pri_count",
        "peak_sum",
    )

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.pulse_count = 0
        self.last_start_sample = 0
        self.pri_sum_seconds = 0.0
        self.pri_count = 0
        self.peak_sum = 0.0


class Deinterleaver:
    def __init__(self, sample_rate_hz: float, pri_tolerance_seconds: float):
        self._sample_rate_hz = sample_rate_hz
        self._pri_tolerance_seconds = pri_tolerance_seconds
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def process(
        self, batch: "pulse_pb2.PulseEventBatch", out: "pulse_pb2.DeinterleaveSummary"
    ) -> None:
        sample_rate_hz = self._sample_rate_hz
        pri_tolerance_seconds = self._pri_tolerance_seconds
        tracks = self._tracks
        next_track_id = self._next_track_id

        for event in batch.events:
            start_sample = event.start_sample
            peak_amplitude = event.peak_amplitude
            pulse_time = start_sample / sample_rate_hz

            # Prefer the closest track whose predicted next-pulse time
            # falls within tolerance; fall back to a track that only has
            # one pulse so far (no PRI to test against yet) if no
            # established track matches.
            best_match = None
            best_diff = pri_tolerance_seconds
            seed_match = None

            for track in tracks:
                if track.pri_count > 0:
                    last_time = track.last_start_sample / sample_rate_hz
                    predicted = last_time + (track.pri_sum_seconds / track.pri_count)
                    diff = abs(predicted - pulse_time)
                    if diff <= best_diff:
                        best_diff = diff
                        best_match = track
                elif track.pulse_count == 1 and seed_match is None:
                    seed_match = track

            target = best_match if best_match is not None else seed_match
            if target is None:
                target = _Track(next_track_id)
                next_track_id += 1
                tracks.append(target)

            if target.pulse_count > 0:
                last_time = target.last_start_sample / sample_rate_hz
                target.pri_sum_seconds += pulse_time - last_time
                target.pri_count += 1
            target.last_start_sample = start_sample
            target.peak_sum += peak_amplitude
            target.pulse_count += 1

        self._next_track_id = next_track_id

        out.Clear()
        for track in tracks:
            t = out.tracks.add()
            t.track_id = track.track_id
            t.pulse_count = track.pulse_count
            t.estimated_pri_seconds = (
                track.pri_sum_seconds / track.pri_count if track.pri_count > 0 else 0.0
            )
            t.mean_peak_amplitude = (
                track.peak_sum / track.pulse_count if track.pulse_count > 0 else 0.0
            )
