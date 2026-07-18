#!/usr/bin/env python3
"""jammer_service.py: standalone Python port of
microservice/jammer_service/main.cpp, running the exact same
pulsecore.jammer.JammerDetector used by monolith/stages.py's JammerStage.
Third stage of the pipeline chain (detector -> spectrogram -> jammer ->
stats -> deinterleaver): connects out to stats_service (the next stage)
at startup, then listens for spectrogram_service.

This is the last of the three stages that read frame.iq (the largest
field by far -- 10,000 samples/batch), which is exactly why detector,
spectrogram, and jammer are grouped first in the chain: once this stage
is done with it, nothing downstream (stats_service, deinterleave_service)
ever reads it again, so it's cleared right here before forwarding instead
of being serialized and transmitted two more times for no reason.

    jammer_service.py [listen_port] [next_host] [next_port]
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.jammer import JammerDetector
from microservice import framing

_POWER_THRESHOLD = 20.0
_DUTY_CYCLE_THRESHOLD = 0.5


def main() -> int:
    listen_port = int(sys.argv[1]) if len(sys.argv) > 1 else 50053
    next_host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    next_port = int(sys.argv[3]) if len(sys.argv) > 3 else 50054

    print(f"[jammer_service.py] connecting to stats_service at {next_host}:{next_port}")
    try:
        downstream = framing.connect(next_host, next_port)
    except OSError:
        print(
            "[jammer_service.py] failed to connect (is stats_service running?)", file=sys.stderr
        )
        return 1

    listener = framing.listen(listen_port)
    print(
        f"[jammer_service.py] listening on 127.0.0.1:{listen_port}, "
        "waiting for spectrogram_service..."
    )
    upstream = framing.accept(listener)
    print("[jammer_service.py] spectrogram_service connected")

    detector = JammerDetector(
        power_threshold=_POWER_THRESHOLD, duty_cycle_threshold=_DUTY_CYCLE_THRESHOLD
    )
    frame = pulse_pb2.PipelineFrame()
    # Kept separately from frame because frame.jam gets cleared before
    # every forward (see below) -- this is what gets printed after the
    # loop ends.
    last_summary = pulse_pb2.JamSummary()
    frames_forwarded = 0
    # See spectrogram_service.py for why the timer starts on the first
    # successful receive rather than before the loop.
    steady_state_start = None
    steady_state_ms = 0.0

    while True:
        payload = framing.recv_message(upstream)
        if payload is None:
            break
        if steady_state_start is None:
            steady_state_start = time.perf_counter()
        frame.ParseFromString(payload)

        detector.process(frame.iq, frame.jam)
        last_summary.CopyFrom(frame.jam)

        # Nothing downstream (stats_service, deinterleave_service) reads
        # frame.iq or frame.jam -- this is the last stage that needs the
        # raw samples, so drop them here instead of paying to move the
        # biggest message in the pipeline across two more hops unused.
        frame.ClearField("iq")
        frame.ClearField("jam")

        if not framing.send_message(downstream, frame.SerializeToString()):
            print(
                "[jammer_service.py] forward failed, stats_service may have exited",
                file=sys.stderr,
            )
            break
        frames_forwarded += 1

    if steady_state_start is not None:
        steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    print(f"[jammer_service.py] received/forwarded {frames_forwarded} frame(s) over TCP")
    print(f"[jammer_service.py] STEADY_STATE_MS {steady_state_ms:.6f}")
    print(
        f"[jammer_service.py] {last_summary.batches_flagged}/{last_summary.batches_total} "
        f"batches flagged, max_duty_cycle={last_summary.max_duty_cycle:.3f} "
        f"max_mean_power={last_summary.max_mean_power:.2f}"
    )

    upstream.close()
    downstream.close()
    listener.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
