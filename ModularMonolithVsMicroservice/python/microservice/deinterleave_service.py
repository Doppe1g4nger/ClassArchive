#!/usr/bin/env python3
"""deinterleave_service.py: standalone Python port of
microservice/deinterleave_service/main.cpp, running the exact same
pulsecore.deinterleaver.Deinterleaver used by monolith/stages.py's
DeinterleaverStage. Fifth and final stage of the pipeline chain (detector
-> spectrogram -> jammer -> stats -> deinterleaver): it's the chain's
sink, so it only listens for stats_service and never forwards -- it folds
frame.events into candidate emitter tracks and prints the result once the
upstream connection closes.

As the chain's sink, this process is also where the pipeline's overall
steady-state measurement is taken: the time from its first successful
receive to its last is, by construction, the time it took the whole
pipeline to drain once fully connected, with zero cross-process timestamp
correlation required -- no data can reach this process until every
upstream hop has finished connecting. See detector_service.py, whose own
connect() can't succeed any earlier than that either.

    deinterleave_service.py [listen_port]
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.deinterleaver import Deinterleaver
from microservice import framing

_SAMPLE_RATE_HZ = 10_000_000.0
_PRI_TOLERANCE_SECONDS = 1e-7


def main() -> int:
    listen_port = int(sys.argv[1]) if len(sys.argv) > 1 else 50055

    listener = framing.listen(listen_port)
    print(
        f"[deinterleave_service.py] listening on 127.0.0.1:{listen_port}, "
        "waiting for stats_service..."
    )
    upstream = framing.accept(listener)
    print("[deinterleave_service.py] stats_service connected")

    deinterleaver = Deinterleaver(
        sample_rate_hz=_SAMPLE_RATE_HZ, pri_tolerance_seconds=_PRI_TOLERANCE_SECONDS
    )
    frame = pulse_pb2.PipelineFrame()
    frames_received = 0
    # See spectrogram_service.py for why the timer starts on the first
    # successful receive rather than before the loop -- here that choice
    # is what makes this the pipeline-wide steady-state number (see this
    # module's docstring).
    steady_state_start = None
    steady_state_ms = 0.0

    while True:
        payload = framing.recv_message(upstream)
        if payload is None:
            break
        if steady_state_start is None:
            steady_state_start = time.perf_counter()
        frame.ParseFromString(payload)
        deinterleaver.process(frame.events, frame.deinterleave)
        frames_received += 1

    if steady_state_start is not None:
        steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    print(
        f"[deinterleave_service.py] received {frames_received} frame(s) over TCP, end of chain"
    )
    print(f"[deinterleave_service.py] STEADY_STATE_MS {steady_state_ms:.6f}")
    print(f"[deinterleave_service.py] {len(frame.deinterleave.tracks)} track(s)")
    for track in frame.deinterleave.tracks:
        print(
            f"[deinterleave_service.py]   track {track.track_id}: pulses={track.pulse_count} "
            f"estimated_pri_us={track.estimated_pri_seconds * 1e6:.2f} "
            f"mean_peak={track.mean_peak_amplitude:.3f}"
        )

    upstream.close()
    listener.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
