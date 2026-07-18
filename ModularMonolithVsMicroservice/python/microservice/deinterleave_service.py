#!/usr/bin/env python3
"""deinterleave_service.py: standalone Python port of
microservice/deinterleave_service/main.cpp, running the exact same
pulsecore.deinterleaver.Deinterleaver used by monolith/stages.py's
DeinterleaverStage. Fifth and final stage of the pipeline chain (detector
-> spectrogram -> jammer -> stats -> deinterleaver): it's the chain's
sink, so it only listens for stats_service and never forwards -- it folds
frame.events into candidate emitter tracks and prints the result once the
upstream connection closes.

    deinterleave_service.py [listen_port]
"""
import sys
import os

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

    while True:
        payload = framing.recv_message(upstream)
        if payload is None:
            break
        frame.ParseFromString(payload)
        deinterleaver.process(frame.events, frame.deinterleave)
        frames_received += 1

    print(
        f"[deinterleave_service.py] received {frames_received} frame(s) over TCP, end of chain"
    )
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
