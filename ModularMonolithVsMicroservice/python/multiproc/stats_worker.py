"""stats_worker.py: forked-process port of microservice/stats_service.py
for the multiproc build. See detector_worker.py for the pipe/sentinel
notes.
"""
import time

from pulsecore import pulse_pb2
from pulsecore.pulse_stats import PulseStatsAccumulator

_SAMPLE_RATE_HZ = 10_000_000.0


def run(conn_in, conn_out, results) -> None:
    accumulator = PulseStatsAccumulator(sample_rate_hz=_SAMPLE_RATE_HZ)
    frame = pulse_pb2.PipelineFrame()
    last_summary = pulse_pb2.PulseSummary()
    frames_forwarded = 0
    steady_state_start = None
    steady_state_ms = 0.0

    while True:
        payload = conn_in.recv()
        if payload is None:
            break
        if steady_state_start is None:
            steady_state_start = time.perf_counter()
        frame.ParseFromString(payload)

        accumulator.add(frame.events)
        frame.stats.CopyFrom(accumulator.finalize())
        last_summary.CopyFrom(frame.stats)

        # deinterleave_worker (next hop) only reads frame.events; nothing
        # downstream of it ever reads frame.stats.
        frame.ClearField("stats")

        conn_out.send(frame.SerializeToString())
        frames_forwarded += 1

    if steady_state_start is not None:
        steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0
    conn_out.send(None)
    conn_in.close()
    conn_out.close()

    results.put((
        "stats",
        {
            "frames_forwarded": frames_forwarded,
            "steady_state_ms": steady_state_ms,
            "pulse_count": last_summary.pulse_count,
            "mean_peak_amplitude": last_summary.mean_peak_amplitude,
            "mean_duration_seconds": last_summary.mean_duration_seconds,
            "mean_pri_seconds": last_summary.mean_pri_seconds,
            "min_peak_amplitude": last_summary.min_peak_amplitude,
            "max_peak_amplitude": last_summary.max_peak_amplitude,
        },
    ))
