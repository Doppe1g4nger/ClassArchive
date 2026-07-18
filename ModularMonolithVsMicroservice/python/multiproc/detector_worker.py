"""detector_worker.py: the multiproc build's producer, run in its own
forked process by multiproc_monolith_app.py. Same job as
microservice/detector_service.py's main() -- generate the synthetic IQ
stream, run detection locally, and hand the frame to the next stage --
except "the next stage" is reached over a multiprocessing.Connection (one
end of a pipe the parent created before forking any worker, see
multiproc_monolith_app.py) instead of a TCP socket, so there's no
connect() to wait on: the pipe already exists, synchronously, before this
process is even spawned.
"""
import time

from pulsecore import pulse_pb2
from pulsecore.iq_source import SyntheticIQSource
from pulsecore.pulse_detector import PulseDetector

_SAMPLE_RATE_HZ = 10_000_000.0


def run(conn_out, num_pulses: int, results) -> None:
    detector = PulseDetector(amplitude_threshold=6.0, sample_rate_hz=_SAMPLE_RATE_HZ)
    source = SyntheticIQSource(sample_rate_hz=_SAMPLE_RATE_HZ, num_pulses=num_pulses)

    # Reused across iterations for the same reason the other builds do --
    # frame.iq is filled directly by next_batch() below (no copy).
    frame = pulse_pb2.PipelineFrame()
    batches_sent = 0

    # No connect() to wait behind -- conn_out is already a live pipe by
    # the time this function starts running, so the timer can start right
    # at the top of the loop with no setup cost to exclude.
    steady_state_start = time.perf_counter()
    while source.next_batch(frame.iq):
        frame.events.Clear()
        detector.process(frame.iq, frame.events)
        conn_out.send(frame.SerializeToString())
        batches_sent += 1
    steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0

    # Explicit end-of-stream sentinel rather than relying on the pipe's
    # EOF/close semantics: every worker here is forked from the same
    # parent after all four pipes already exist, so every sibling process
    # inherits its own duplicate file descriptor for every pipe end, not
    # just the one or two it actually uses. Reaching real EOF on a pipe
    # requires *every* copy of its write end to be closed, which would
    # mean every worker explicitly closing ends it never touches -- a
    # None sentinel sidesteps that bookkeeping entirely. Contrast with
    # the TCP build's framing.recv_message(), which can rely on socket
    # half-close because each socket only ever has the two processes on
    # its two ends.
    conn_out.send(None)
    conn_out.close()

    results.put(("detector", {"batches_sent": batches_sent, "steady_state_ms": steady_state_ms}))
