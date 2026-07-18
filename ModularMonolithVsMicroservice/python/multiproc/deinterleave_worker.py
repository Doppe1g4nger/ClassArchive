"""deinterleave_worker.py: forked-process port of
microservice/deinterleave_service.py -- this build's sink, and where the
pipeline-wide steady-state number is measured, for a simpler version of
the argument that module makes: no batch can reach this process until it
has flowed through every upstream worker, so this process's own
first-recv-to-last-recv span is, by construction, the whole pipeline's
steady-state drain time. Simpler here because there's no connection-setup
race to reason about at all (see detector_worker.py) -- these are the
same pipe endpoints, already fully wired by the parent before any worker
was even forked.
"""
import time

from pulsecore import pulse_pb2
from pulsecore.deinterleaver import Deinterleaver

_SAMPLE_RATE_HZ = 10_000_000.0
_PRI_TOLERANCE_SECONDS = 1e-7


def run(conn_in, results) -> None:
    deinterleaver = Deinterleaver(
        sample_rate_hz=_SAMPLE_RATE_HZ, pri_tolerance_seconds=_PRI_TOLERANCE_SECONDS
    )
    frame = pulse_pb2.PipelineFrame()
    frames_received = 0
    steady_state_start = None
    steady_state_ms = 0.0

    while True:
        payload = conn_in.recv()
        if payload is None:
            break
        if steady_state_start is None:
            steady_state_start = time.perf_counter()
        frame.ParseFromString(payload)
        deinterleaver.process(frame.events, frame.deinterleave)
        frames_received += 1

    if steady_state_start is not None:
        steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0
    conn_in.close()

    tracks = [
        {
            "track_id": t.track_id,
            "pulse_count": t.pulse_count,
            "estimated_pri_seconds": t.estimated_pri_seconds,
            "mean_peak_amplitude": t.mean_peak_amplitude,
        }
        for t in frame.deinterleave.tracks
    ]
    results.put((
        "deinterleave",
        {
            "frames_received": frames_received,
            "steady_state_ms": steady_state_ms,
            "tracks": tracks,
        },
    ))
