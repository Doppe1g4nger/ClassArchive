"""Array-native stats + deinterleaver for the numpy variant
(theoretical-limits branch, round three).

Profiling round three found this variant's two biggest remaining costs
were the two stages it had never ported: pulsecore's pure-Python
PulseStatsAccumulator and Deinterleaver, fed through a per-batch
protobuf PulseEventBatch built solely to satisfy their signatures.
These replacements take the detector kernel's raw event arrays
directly -- no protobuf anywhere on the per-batch path.

PulseStatsArrays is genuinely vectorized, and one part got *better*
than vectorized: the sum of consecutive-start PRI gaps telescopes --
sum((s[k]-s[k-1])/rate) == (s[last]-s[first])/rate exactly, in integer
sample indices -- so the whole intra-batch PRI accumulation is O(1).
(The division-per-gap rounding the reference does is gone, which is a
tolerance-level difference; this branch retired bit-exactness rounds
ago.)

DeinterleaverArrays keeps the reference's sequential per-event loop --
track assignment is inherently order-dependent (each event's match
depends on track means updated by the previous event), so "vectorize"
would mean "different algorithm, different results". What it sheds is
everything around the loop: protobuf field access (tolist() up front,
native floats/ints in the loop) and the per-batch DeinterleaveSummary
rebuild -- the summary is built once, at the end, because nothing
reads the intermediate ones.
"""
from pulsecore import pulse_pb2


class PulseStatsArrays:
    """Vectorized drop-in for pulsecore.pulse_stats.PulseStatsAccumulator,
    fed event arrays instead of a PulseEventBatch."""

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

    def add(self, ev_start, ev_peak, ev_dur) -> None:
        n = int(ev_start.shape[0])
        if n == 0:
            return
        self._count += n
        self._peak_sum += float(ev_peak.sum())
        self._duration_sum += float(ev_dur.sum())
        batch_min = float(ev_peak.min())
        batch_max = float(ev_peak.max())
        if batch_min < self._peak_min:
            self._peak_min = batch_min
        if batch_max > self._peak_max:
            self._peak_max = batch_max

        first = int(ev_start[0])
        last = int(ev_start[-1])
        if self._have_prev_start:
            self._pri_sum += (first - self._prev_start_sample) / self._sample_rate_hz
            self._pri_count += 1
        # Telescoped: the n-1 intra-batch gaps sum to last-first exactly.
        self._pri_sum += (last - first) / self._sample_rate_hz
        self._pri_count += n - 1
        self._prev_start_sample = last
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


class _Track:
    __slots__ = ("track_id", "pulse_count", "last_start_sample", "pri_sum_seconds",
                 "pri_count", "peak_sum")

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.pulse_count = 0
        self.last_start_sample = 0
        self.pri_sum_seconds = 0.0
        self.pri_count = 0
        self.peak_sum = 0.0


class DeinterleaverArrays:
    """Array-fed port of pulsecore.deinterleaver.Deinterleaver -- same
    sequential assignment algorithm, no protobuf on the per-batch path,
    summary built once at the end via summary()."""

    def __init__(self, sample_rate_hz: float, pri_tolerance_seconds: float):
        self._sample_rate_hz = sample_rate_hz
        self._pri_tolerance_seconds = pri_tolerance_seconds
        self._tracks: list[_Track] = []
        self._next_track_id = 1

    def process(self, ev_start, ev_peak) -> None:
        sample_rate_hz = self._sample_rate_hz
        pri_tolerance_seconds = self._pri_tolerance_seconds
        tracks = self._tracks
        next_track_id = self._next_track_id

        starts = ev_start.tolist()
        peaks = ev_peak.tolist()
        for e in range(len(starts)):
            start_sample = starts[e]
            peak_amplitude = peaks[e]
            pulse_time = start_sample / sample_rate_hz

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

    def summary(self) -> "pulse_pb2.DeinterleaveSummary":
        out = pulse_pb2.DeinterleaveSummary()
        for track in self._tracks:
            t = out.tracks.add()
            t.track_id = track.track_id
            t.pulse_count = track.pulse_count
            t.estimated_pri_seconds = (
                track.pri_sum_seconds / track.pri_count if track.pri_count > 0 else 0.0
            )
            t.mean_peak_amplitude = (
                track.peak_sum / track.pulse_count if track.pulse_count > 0 else 0.0
            )
        return out
