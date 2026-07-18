#!/usr/bin/env python3
"""stats_service.py: standalone Python port of
microservice/stats_service/main.cpp, running the exact same
pulsecore.pulse_stats.PulseStatsAccumulator used by monolith/stages.py's
StatsStage. Fourth stage of the pipeline chain (detector -> spectrogram ->
jammer -> stats -> deinterleaver): connects out to deinterleave_service
(the next and final stage) at startup, then listens for jammer_service. By
this point frame.iq has already been cleared upstream (see
jammer_service.py) since nothing from here on needs it. For each frame:
folds frame.events into running stats, and forwards the frame downstream
with its own stats field cleared before sending -- deinterleave_service
doesn't read it.

    stats_service.py [listen_port] [next_host] [next_port]
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.pulse_stats import PulseStatsAccumulator
from microservice import framing

_SAMPLE_RATE_HZ = 10_000_000.0


def main() -> int:
    listen_port = int(sys.argv[1]) if len(sys.argv) > 1 else 50054
    next_host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    next_port = int(sys.argv[3]) if len(sys.argv) > 3 else 50055

    print(f"[stats_service.py] connecting to deinterleave_service at {next_host}:{next_port}")
    try:
        downstream = framing.connect(next_host, next_port)
    except OSError:
        print(
            "[stats_service.py] failed to connect (is deinterleave_service running?)",
            file=sys.stderr,
        )
        return 1

    listener = framing.listen(listen_port)
    print(
        f"[stats_service.py] listening on 127.0.0.1:{listen_port}, waiting for jammer_service..."
    )
    upstream = framing.accept(listener)
    print("[stats_service.py] jammer_service connected")

    accumulator = PulseStatsAccumulator(sample_rate_hz=_SAMPLE_RATE_HZ)
    frame = pulse_pb2.PipelineFrame()
    # Kept separately from frame because frame.stats gets cleared before
    # every forward (see below) -- this is what gets printed after the
    # loop ends.
    last_summary = pulse_pb2.PulseSummary()
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

        accumulator.add(frame.events)
        frame.stats.CopyFrom(accumulator.finalize())
        last_summary.CopyFrom(frame.stats)

        # deinterleave_service (next hop) only reads frame.events;
        # nothing downstream of it ever reads frame.stats.
        frame.ClearField("stats")

        if not framing.send_message(downstream, frame.SerializeToString()):
            print(
                "[stats_service.py] forward failed, deinterleave_service may have exited",
                file=sys.stderr,
            )
            break
        frames_forwarded += 1

    if steady_state_start is not None:
        steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    print(f"[stats_service.py] received/forwarded {frames_forwarded} frame(s) over TCP")
    print(f"[stats_service.py] STEADY_STATE_MS {steady_state_ms:.6f}")
    print(
        f"[stats_service.py] pulses={last_summary.pulse_count} "
        f"mean_peak={last_summary.mean_peak_amplitude:.3f} "
        f"mean_dur_us={last_summary.mean_duration_seconds * 1e6:.2f} "
        f"mean_pri_us={last_summary.mean_pri_seconds * 1e6:.2f} "
        f"min_peak={last_summary.min_peak_amplitude:.3f} "
        f"max_peak={last_summary.max_peak_amplitude:.3f}"
    )

    upstream.close()
    downstream.close()
    listener.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
