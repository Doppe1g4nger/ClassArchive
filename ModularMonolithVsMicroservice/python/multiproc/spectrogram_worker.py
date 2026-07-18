"""spectrogram_worker.py: forked-process port of
microservice/spectrogram_service.py for the multiproc build. Same
algorithm and the same field-clearing-before-forward discipline, just
reading from and writing to multiprocessing.Connection pipe ends instead
of TCP sockets -- see detector_worker.py for why that means no
connection-setup wait, and for why end-of-stream is an explicit None
sentinel instead of relying on pipe EOF.
"""
import time

from pulsecore import pulse_pb2
from pulsecore.spectrogram import SpectrogramAnalyzer

_SAMPLE_RATE_HZ = 10_000_000.0
_NUM_BINS = 8


def run(conn_in, conn_out, results) -> None:
    analyzer = SpectrogramAnalyzer(sample_rate_hz=_SAMPLE_RATE_HZ, num_bins=_NUM_BINS)
    frame = pulse_pb2.PipelineFrame()
    last_summary = pulse_pb2.SpectrogramSummary()
    frames_forwarded = 0
    # Set on the first successful receive, not before -- the wait for
    # that first message is this stage's share of the pipeline's fill
    # time, which we want excluded from a steady-state measurement, same
    # reasoning as spectrogram_service.py.
    steady_state_start = None
    steady_state_ms = 0.0

    while True:
        payload = conn_in.recv()
        if payload is None:
            break
        if steady_state_start is None:
            steady_state_start = time.perf_counter()
        frame.ParseFromString(payload)

        analyzer.process(frame.iq, frame.spectrogram)
        last_summary.CopyFrom(frame.spectrogram)

        # jammer_worker (next hop) only reads frame.iq; nothing
        # downstream of it ever reads frame.spectrogram.
        frame.ClearField("spectrogram")

        conn_out.send(frame.SerializeToString())
        frames_forwarded += 1

    if steady_state_start is not None:
        steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0
    conn_out.send(None)
    conn_in.close()
    conn_out.close()

    results.put((
        "spectrogram",
        {
            "frames_forwarded": frames_forwarded,
            "steady_state_ms": steady_state_ms,
            "bin_hz": last_summary.bin_hz,
            "frame_count": last_summary.frame_count,
            "max_magnitude": list(last_summary.max_magnitude),
            "mean_magnitude": list(last_summary.mean_magnitude),
        },
    ))
