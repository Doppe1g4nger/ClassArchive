#!/usr/bin/env python3
"""detector_service.py: standalone Python port of
microservice/detector_service/main.cpp, running the exact same
pulsecore.pulse_detector.PulseDetector used by monolith/stages.py's
DetectorStage. First stage of the pipeline chain (detector -> spectrogram
-> jammer -> stats -> deinterleaver): it's the chain's pure producer, so
it never listens -- it generates the synthetic IQ stream, runs detection
locally, and connects out to spectrogram_service (the next stage) as a
plain TCP client, streaming one serialized PipelineFrame per batch.

    detector_service.py [host] [port] [num_pulses]
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.iq_source import SyntheticIQSource
from pulsecore.pulse_detector import PulseDetector
from microservice import framing

_SAMPLE_RATE_HZ = 10_000_000.0


def main() -> int:
    next_host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    next_port = int(sys.argv[2]) if len(sys.argv) > 2 else 50051
    # Default of 1000 pulses is exactly one buffer's worth at this repo's
    # 1,000,000-pulse/sec, 1000-microsecond-buffer scale.
    num_pulses = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    print(f"[detector_service.py] connecting to spectrogram_service at {next_host}:{next_port}")
    try:
        downstream = framing.connect(next_host, next_port)
    except OSError:
        print(
            "[detector_service.py] failed to connect (is spectrogram_service running?)",
            file=sys.stderr,
        )
        return 1

    detector = PulseDetector(amplitude_threshold=6.0, sample_rate_hz=_SAMPLE_RATE_HZ)
    source = SyntheticIQSource(sample_rate_hz=_SAMPLE_RATE_HZ, num_pulses=num_pulses)

    # Reused across iterations for the same reason the C++ services do --
    # frame.iq is filled directly by next_batch() below (no copy).
    frame = pulse_pb2.PipelineFrame()
    batches_sent = 0

    # Timed region starts right after connect() succeeds -- which, thanks
    # to this chain's reverse-order startup (see
    # scripts/run_python_microservices.sh), can only happen once every
    # downstream hop is already listening. So this measurement excludes
    # not just this process's own connection setup but the whole chain's.
    # See microservice/deinterleave_service.py for the matching
    # measurement at the other end of the pipeline.
    steady_state_start = time.perf_counter()
    while source.next_batch(frame.iq):
        frame.events.Clear()
        detector.process(frame.iq, frame.events)

        if not framing.send_message(downstream, frame.SerializeToString()):
            print(
                "[detector_service.py] send failed, spectrogram_service may have exited",
                file=sys.stderr,
            )
            downstream.close()
            return 1
        batches_sent += 1
    steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    print(f"[detector_service.py] streamed {batches_sent} frame(s) into the chain, closing")
    print(f"[detector_service.py] STEADY_STATE_MS {steady_state_ms:.6f}")
    downstream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
