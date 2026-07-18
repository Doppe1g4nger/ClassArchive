#!/usr/bin/env python3
"""spectrogram_service.py: standalone Python port of
microservice/spectrogram_service/main.cpp, running the exact same
pulsecore.spectrogram.SpectrogramAnalyzer used by monolith/stages.py's
SpectrogramStage. Second stage of the pipeline chain (detector ->
spectrogram -> jammer -> stats -> deinterleaver): connects out to
jammer_service (the next stage) at startup, then listens for
detector_service. For each frame it receives: folds frame.iq into a
running magnitude spectrum, and forwards the frame downstream with its
own spectrogram field cleared before sending -- jammer_service doesn't
read it, so there's no reason to pay to serialize and transmit it.

    spectrogram_service.py [listen_port] [next_host] [next_port]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.spectrogram import SpectrogramAnalyzer
from microservice import framing

_SAMPLE_RATE_HZ = 10_000_000.0
_NUM_BINS = 8


def main() -> int:
    listen_port = int(sys.argv[1]) if len(sys.argv) > 1 else 50052
    next_host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    next_port = int(sys.argv[3]) if len(sys.argv) > 3 else 50053

    print(f"[spectrogram_service.py] connecting to jammer_service at {next_host}:{next_port}")
    try:
        downstream = framing.connect(next_host, next_port)
    except OSError:
        print(
            "[spectrogram_service.py] failed to connect (is jammer_service running?)",
            file=sys.stderr,
        )
        return 1

    listener = framing.listen(listen_port)
    print(
        f"[spectrogram_service.py] listening on 127.0.0.1:{listen_port}, "
        "waiting for detector_service..."
    )
    upstream = framing.accept(listener)
    print("[spectrogram_service.py] detector_service connected")

    analyzer = SpectrogramAnalyzer(sample_rate_hz=_SAMPLE_RATE_HZ, num_bins=_NUM_BINS)
    frame = pulse_pb2.PipelineFrame()
    # Kept separately from frame because frame.spectrogram gets cleared
    # before every forward (see below) -- this is what gets printed after
    # the loop ends.
    last_summary = pulse_pb2.SpectrogramSummary()
    frames_forwarded = 0

    while True:
        payload = framing.recv_message(upstream)
        if payload is None:
            break
        frame.ParseFromString(payload)

        analyzer.process(frame.iq, frame.spectrogram)
        last_summary.CopyFrom(frame.spectrogram)

        # jammer_service (next hop) only reads frame.iq; nothing
        # downstream of it ever reads frame.spectrogram, so there's no
        # reason to keep paying to serialize and transmit it past here.
        frame.ClearField("spectrogram")

        if not framing.send_message(downstream, frame.SerializeToString()):
            print(
                "[spectrogram_service.py] forward failed, jammer_service may have exited",
                file=sys.stderr,
            )
            break
        frames_forwarded += 1

    print(f"[spectrogram_service.py] received/forwarded {frames_forwarded} frame(s) over TCP")
    print(
        f"[spectrogram_service.py] {len(last_summary.max_magnitude)} bins, "
        f"{last_summary.bin_hz:.1f} Hz spacing, {last_summary.frame_count} frames"
    )
    for i in range(len(last_summary.max_magnitude)):
        bin_center = (i + 0.5) * last_summary.bin_hz
        print(
            f"[spectrogram_service.py]   bin {i} (~{bin_center:.0f} Hz): "
            f"max={last_summary.max_magnitude[i]:.3f} mean={last_summary.mean_magnitude[i]:.3f}"
        )

    upstream.close()
    downstream.close()
    listener.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
