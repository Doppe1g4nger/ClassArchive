#!/usr/bin/env python3
"""multiproc_monolith_app.py: a third architecture, between the other
two. Answers a question "Python vs C++" (see the top-level README)
raises but can't settle on its own: the microservices build beats the
Python monolith at full scale because it's five separate OS processes,
each with its own GIL, free to run on different cores -- but is escaping
the GIL what actually matters, or does the network/serialization
boundary matter too? This build has multiple processes and no network
boundary at all, to isolate that variable.

Structurally it's much closer to the modular monolith than to the
microservices build: one script, one command, no ports, no independent
per-stage executables that have to be started in a particular order. It
gets there by using Python's multiprocessing module to fork five worker
processes directly -- detector_worker.py through deinterleave_worker.py,
each mirroring its microservice/*_service.py counterpart's algorithm and
field-clearing discipline exactly -- and wiring them together with
multiprocessing.Pipe() connections instead of TCP sockets.

That substitution removes two costs the TCP build has to pay without
changing anything about the actual pipeline logic:

- No connection-setup dance. A Pipe() is created synchronously by this
  process before any worker is even spawned, so there's no
  bind()/listen()/accept()/connect() sequence, no reverse-of-data-flow
  startup order, and no wait_for_port polling -- compare
  scripts/run_python_microservices.sh, none of which this build needs.
- Cheaper process startup. multiprocessing's default start method on
  Linux is fork(), which clones this already-running interpreter
  (already having imported protobuf, pulsecore, etc.) instead of
  spawning five fresh `python3` processes that each redo interpreter
  startup and imports from scratch. This build asks for "fork" by name
  rather than relying on the platform default, since that cheap startup
  is part of the point being tested here.

What it does *not* remove is the GIL-escaping property that made the
microservices build win at full scale in the first place: five separate
OS processes still means five separate GILs, so the pipeline can still
use multiple cores concurrently -- see the README's steady-state
benchmark for whether that's enough on its own, or whether the TCP
build's network boundary was buying it something extra (or costing it
something) beyond that.

    multiproc_monolith_app.py [num_pulses]
"""
import sys
import os
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from multiproc import (
    detector_worker,
    spectrogram_worker,
    jammer_worker,
    stats_worker,
    deinterleave_worker,
)


def main() -> int:
    # Default of 1000 pulses matches every other build in this repo.
    num_pulses = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    ctx = multiprocessing.get_context("fork")
    results = ctx.Queue()

    # One duplex=False Pipe per link, matching the chain's data-flow
    # order exactly: detector -> spectrogram -> jammer -> stats ->
    # deinterleave. All four exist before any worker is forked, which is
    # exactly why every worker signals end-of-stream with an explicit
    # None sentinel instead of relying on pipe EOF -- see
    # detector_worker.py's docstring.
    det_to_spec_recv, det_to_spec_send = ctx.Pipe(duplex=False)
    spec_to_jam_recv, spec_to_jam_send = ctx.Pipe(duplex=False)
    jam_to_stats_recv, jam_to_stats_send = ctx.Pipe(duplex=False)
    stats_to_dei_recv, stats_to_dei_send = ctx.Pipe(duplex=False)

    processes = [
        ctx.Process(target=detector_worker.run, args=(det_to_spec_send, num_pulses, results)),
        ctx.Process(
            target=spectrogram_worker.run, args=(det_to_spec_recv, spec_to_jam_send, results)
        ),
        ctx.Process(target=jammer_worker.run, args=(spec_to_jam_recv, jam_to_stats_send, results)),
        ctx.Process(target=stats_worker.run, args=(jam_to_stats_recv, stats_to_dei_send, results)),
        ctx.Process(target=deinterleave_worker.run, args=(stats_to_dei_recv, results)),
    ]
    for p in processes:
        p.start()

    # This process's own copies of every pipe end are no longer needed --
    # each worker already inherited its own copies via fork(). Closing
    # them here doesn't affect the workers (fork() gave each of them an
    # independent file descriptor), it just stops this process from
    # holding pipes open indefinitely while it waits below.
    for conn in (
        det_to_spec_recv,
        det_to_spec_send,
        spec_to_jam_recv,
        spec_to_jam_send,
        jam_to_stats_recv,
        jam_to_stats_send,
        stats_to_dei_recv,
        stats_to_dei_send,
    ):
        conn.close()

    # Drain the results queue before join()ing -- a child that has put
    # more data into a Queue than fits in its OS pipe buffer will block
    # on that put() until a consumer reads it, so joining first can
    # deadlock. Every worker puts exactly one (stage_name, payload) pair.
    stage_results = {}
    for _ in range(len(processes)):
        stage, payload = results.get()
        stage_results[stage] = payload

    for p in processes:
        p.join()

    dei = stage_results["deinterleave"]
    print(
        f"[multiproc_monolith_app.py] processed {dei['frames_received']} IQ batches through a "
        f"forked chain of {len(processes)} worker processes"
    )
    print(f"[multiproc_monolith_app.py] STEADY_STATE_MS {dei['steady_state_ms']:.6f}")

    spec = stage_results["spectrogram"]
    print(
        f"[multiproc_monolith_app.py] spectrogram: {len(spec['max_magnitude'])} bins, "
        f"{spec['bin_hz']:.1f} Hz spacing, {spec['frame_count']} frames"
    )
    for i, (max_mag, mean_mag) in enumerate(zip(spec["max_magnitude"], spec["mean_magnitude"])):
        bin_center = (i + 0.5) * spec["bin_hz"]
        print(
            f"[multiproc_monolith_app.py]   bin {i} (~{bin_center:.0f} Hz): "
            f"max={max_mag:.3f} mean={mean_mag:.3f}"
        )

    jam = stage_results["jammer"]
    print(
        f"[multiproc_monolith_app.py] jammer: {jam['batches_flagged']}/{jam['batches_total']} "
        f"batches flagged, max_duty_cycle={jam['max_duty_cycle']:.3f} "
        f"max_mean_power={jam['max_mean_power']:.2f}"
    )

    stats = stage_results["stats"]
    print(
        f"[multiproc_monolith_app.py] stats: pulses={stats['pulse_count']} "
        f"mean_peak={stats['mean_peak_amplitude']:.3f} "
        f"mean_dur_us={stats['mean_duration_seconds'] * 1e6:.2f} "
        f"mean_pri_us={stats['mean_pri_seconds'] * 1e6:.2f} "
        f"min_peak={stats['min_peak_amplitude']:.3f} max_peak={stats['max_peak_amplitude']:.3f}"
    )

    print(f"[multiproc_monolith_app.py] deinterleaver: {len(dei['tracks'])} track(s)")
    for track in dei["tracks"]:
        print(
            f"[multiproc_monolith_app.py]   track {track['track_id']}: "
            f"pulses={track['pulse_count']} "
            f"estimated_pri_us={track['estimated_pri_seconds'] * 1e6:.2f} "
            f"mean_peak={track['mean_peak_amplitude']:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
