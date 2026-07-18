"""Stage wrappers -- Python port of monolith/plugins/*.cpp. Each class
wraps one pulsecore algorithm behind a process(frame) call that reads
whatever pulse::PipelineFrame fields it needs and writes its own field
back, exactly matching the corresponding *_plugin.cpp's
pulse_stage_process() function.

There's no dlopen()/dlsym() step here: Python's import statement already
gives each stage its own separately authored, independently loadable
module, which is this language's version of "dynamically loaded" --
module_api.h's whole C-ABI/dlsym() dance exists to let C++ .so files
loaded at runtime agree on a call signature without sharing source; Python
modules agree on a call signature (every class here exposes the same
process(frame) method) without needing any of that ceremony, because the
interpreter resolves `import` by name at runtime the same way dlopen()
resolves a path by name -- just with far less machinery required to make
it safe.
"""
from pulsecore.deinterleaver import Deinterleaver
from pulsecore.jammer import JammerDetector
from pulsecore.pulse_detector import PulseDetector
from pulsecore.pulse_stats import PulseStatsAccumulator
from pulsecore.spectrogram import SpectrogramAnalyzer


class DetectorStage:
    """First stage of the chain: reads frame.iq (populated by the caller
    before this stage runs) and writes frame.events."""

    def __init__(self, threshold: float, sample_rate_hz: float):
        self._detector = PulseDetector(threshold, sample_rate_hz)

    def process(self, frame) -> None:
        frame.events.Clear()
        self._detector.process(frame.iq, frame.events)


class SpectrogramStage:
    """Second stage: reads frame.iq (populated once, before the chain
    starts) and writes frame.spectrogram."""

    def __init__(self, sample_rate_hz: float, num_bins: int = 8):
        self._analyzer = SpectrogramAnalyzer(sample_rate_hz, num_bins)

    def process(self, frame) -> None:
        self._analyzer.process(frame.iq, frame.spectrogram)


class JammerStage:
    """Third stage: reads frame.iq and writes frame.jam. The last of the
    three raw-sample-consuming stages -- see the microservice build's
    jammer_service.py for where this matters (clearing frame.iq before
    forwarding, since nothing downstream needs it)."""

    def __init__(self, power_threshold: float, duty_cycle_threshold: float):
        self._detector = JammerDetector(power_threshold, duty_cycle_threshold)

    def process(self, frame) -> None:
        self._detector.process(frame.iq, frame.jam)


class StatsStage:
    """Fourth stage: reads frame.events (populated by the detector stage)
    and writes frame.stats."""

    def __init__(self, sample_rate_hz: float):
        self._accumulator = PulseStatsAccumulator(sample_rate_hz)

    def process(self, frame) -> None:
        self._accumulator.add(frame.events)
        frame.stats.CopyFrom(self._accumulator.finalize())


class DeinterleaverStage:
    """Fifth and final stage: reads frame.events and writes
    frame.deinterleave."""

    def __init__(self, sample_rate_hz: float, pri_tolerance_seconds: float):
        self._deinterleaver = Deinterleaver(sample_rate_hz, pri_tolerance_seconds)

    def process(self, frame) -> None:
        self._deinterleaver.process(frame.events, frame.deinterleave)
