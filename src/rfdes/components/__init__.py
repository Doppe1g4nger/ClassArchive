"""Ready-made example RF components.

These illustrate the :class:`~rfdes.component.Component` pattern; build your own
by subclassing ``Component`` and overriding ``on_signal``.
"""

from .active import ADC, Amplifier, Mixer
from .detectors import DetectionFusion, PulseDetector, Spectrogrammer
from .passive import Attenuator, Filter, Splitter
from .sinks import Recorder, SpectrumAnalyzer

__all__ = [
    "Amplifier",
    "Mixer",
    "ADC",
    "Attenuator",
    "Splitter",
    "Filter",
    "PulseDetector",
    "Spectrogrammer",
    "DetectionFusion",
    "Recorder",
    "SpectrumAnalyzer",
]
