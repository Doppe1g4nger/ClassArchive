#!/usr/bin/env python3
"""monolith_app.py: single-process Python port of monolith_main.cpp. Runs
the same five-stage chain -- detector -> spectrogram -> jammer -> stats ->
deinterleaver -- as five imported Python modules instead of five
dlopen()'d .so files, passing one pulse::PipelineFrame through all five
by reference. No serialization here either, for the same reason the C++
monolith doesn't need any: every stage runs in the same process's memory,
so `frame.iq`/`frame.events`/etc. mutations are visible to every stage
without copying anything (see stages.py's docstring for the fuller
version of this point).

CLI differs slightly from monolith_app (the C++ binary): there's no
plugin_dir, since there's nothing to dlopen() -- Python resolves imports
by name, not by a runtime-supplied filesystem path.

    monolith_app.py [num_pulses]
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pulsecore import pulse_pb2
from pulsecore.iq_source import SyntheticIQSource
from monolith.stages import (
    DetectorStage,
    SpectrogramStage,
    JammerStage,
    StatsStage,
    DeinterleaverStage,
)

_SAMPLE_RATE_HZ = 10_000_000.0


def main() -> int:
    # Default of 1000 pulses is exactly one buffer's worth at this repo's
    # 1,000,000-pulse/sec, 1000-microsecond-buffer scale (see
    # pulsecore/iq_source.py).
    num_pulses = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    # Chain order matches monolith_main.cpp / the microservice build
    # exactly (see scripts/run_python_microservices.sh): each stage needs
    # whatever the stages before it in this list have already written
    # into the frame.
    chain = [
        DetectorStage(threshold=6.0, sample_rate_hz=_SAMPLE_RATE_HZ),
        SpectrogramStage(sample_rate_hz=_SAMPLE_RATE_HZ, num_bins=8),
        JammerStage(power_threshold=20.0, duty_cycle_threshold=0.5),
        StatsStage(sample_rate_hz=_SAMPLE_RATE_HZ),
        DeinterleaverStage(sample_rate_hz=_SAMPLE_RATE_HZ, pri_tolerance_seconds=1e-7),
    ]

    source = SyntheticIQSource(sample_rate_hz=_SAMPLE_RATE_HZ, num_pulses=num_pulses)

    # Round three of the theoretical-limits branch: the signal is a
    # GIVEN (real IQ comes from a radio; no architecture choice speeds
    # up the antenna), so every batch is generated before the clock
    # starts and the measured region begins at detection -- the same
    # charter as every other build on this branch. One pre-filled frame
    # per batch; each stage's summary fields are written into the frame
    # the batch flows through, so the LAST frame holds the final
    # summaries to print, exactly like the C++ hosts.
    frames = []
    while True:
        f = pulse_pb2.PipelineFrame()
        if not source.next_batch(f.iq):
            break
        frames.append(f)
    batches = 0

    # Timed region covers detection through deinterleave -- generation
    # (above), imports, and printing the summaries (below) are all
    # excluded, so this number reflects steady-state throughput rather
    # than one-time cost. See microservice/detector_service.py for the
    # equivalent charter on the chain build, and
    # scripts/benchmark_steady_state.sh for how these numbers compare.
    steady_state_start = time.perf_counter()
    for frame in frames:
        for stage in chain:
            stage.process(frame)
        batches += 1
    steady_state_ms = (time.perf_counter() - steady_state_start) * 1000.0
    frame = frames[-1] if frames else pulse_pb2.PipelineFrame()

    print(
        f"[monolith_app.py] processed {batches} IQ batches through an imported chain "
        f"of {len(chain)} modules"
    )
    print(f"[monolith_app.py] STEADY_STATE_MS {steady_state_ms:.6f}")

    spectrogram = frame.spectrogram
    print(
        f"[monolith_app.py] spectrogram: {len(spectrogram.max_magnitude)} bins, "
        f"{spectrogram.bin_hz:.1f} Hz spacing, {spectrogram.frame_count} frames"
    )
    for i in range(len(spectrogram.max_magnitude)):
        bin_center = (i + 0.5) * spectrogram.bin_hz
        print(
            f"[monolith_app.py]   bin {i} (~{bin_center:.0f} Hz): "
            f"max={spectrogram.max_magnitude[i]:.3f} mean={spectrogram.mean_magnitude[i]:.3f}"
        )

    jam = frame.jam
    print(
        f"[monolith_app.py] jammer: {jam.batches_flagged}/{jam.batches_total} batches flagged, "
        f"max_duty_cycle={jam.max_duty_cycle:.3f} max_mean_power={jam.max_mean_power:.2f}"
    )

    stats = frame.stats
    print(
        f"[monolith_app.py] stats: pulses={stats.pulse_count} "
        f"mean_peak={stats.mean_peak_amplitude:.3f} "
        f"mean_dur_us={stats.mean_duration_seconds * 1e6:.2f} "
        f"mean_pri_us={stats.mean_pri_seconds * 1e6:.2f} "
        f"min_peak={stats.min_peak_amplitude:.3f} max_peak={stats.max_peak_amplitude:.3f}"
    )

    tracks = frame.deinterleave
    print(f"[monolith_app.py] deinterleaver: {len(tracks.tracks)} track(s)")
    for track in tracks.tracks:
        print(
            f"[monolith_app.py]   track {track.track_id}: pulses={track.pulse_count} "
            f"estimated_pri_us={track.estimated_pri_seconds * 1e6:.2f} "
            f"mean_peak={track.mean_peak_amplitude:.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
