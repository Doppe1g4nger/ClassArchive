"""jammer_worker.py: forked-process port of
microservice/jammer_service.py for the multiproc build. Also where
frame.iq gets cleared, same as the TCP build -- last stage that reads the
raw samples, so nothing downstream should keep paying to move them. See
detector_worker.py for the pipe/sentinel notes.
"""
import time

from pulsecore import pulse_pb2
from pulsecore.jammer import JammerDetector

_POWER_THRESHOLD = 20.0
_DUTY_CYCLE_THRESHOLD = 0.5


def run(conn_in, conn_out, results) -> None:
    detector = JammerDetector(
        power_threshold=_POWER_THRESHOLD, duty_cycle_threshold=_DUTY_CYCLE_THRESHOLD
    )
    frame = pulse_pb2.PipelineFrame()
    last_summary = pulse_pb2.JamSummary()
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

        detector.process(frame.iq, frame.jam)
        last_summary.CopyFrom(frame.jam)

        # Nothing downstream (stats_worker, deinterleave_worker) reads
        # frame.iq or frame.jam -- same field-clearing discipline as
        # jammer_service.py.
        frame.ClearField("iq")
        frame.ClearField("jam")

        conn_out.send(frame.SerializeToString())
        frames_forwarded += 1

    if steady_state_start is not None:
        steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0
    conn_out.send(None)
    conn_in.close()
    conn_out.close()

    results.put((
        "jammer",
        {
            "frames_forwarded": frames_forwarded,
            "steady_state_ms": steady_state_ms,
            "batches_flagged": last_summary.batches_flagged,
            "batches_total": last_summary.batches_total,
            "max_duty_cycle": last_summary.max_duty_cycle,
            "max_mean_power": last_summary.max_mean_power,
        },
    ))
