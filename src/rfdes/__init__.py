"""rfdes -- component-level discrete-event simulation framework for RF systems.

Compose an RF system from :class:`~rfdes.component.Component` nodes wired
together with ``subscribe`` / ``>>``. The external RF-environment simulator owns
the event queue; this framework places component-processing events onto it via
the :class:`~rfdes.scheduler.Scheduler` Protocol. A standalone
:class:`~rfdes.scheduler.HeapScheduler` lets you run and test systems in
isolation.

Example::

    from rfdes import RFSystem, HeapScheduler
    from rfdes.components import Amplifier, Mixer, ADC

    sys = RFSystem(HeapScheduler())
    lna = sys.add(Amplifier("lna", gain_db=20, processing_delay=1e-9))
    mix = sys.add(Mixer("mix", lo_freq=1e9, processing_delay=2e-9))
    adc = sys.add(ADC("adc", processing_delay=5e-9))
    lna >> mix >> adc
    sys.set_entry(lna)
"""

from . import delays
from .component import Component, MergeComponent
from .datatypes import DetectionReport, PulseBuffer, Spectrogram
from .events import SIGNAL_RX, DEFAULT_IQ_DTYPE, DataObject, Event, SignalPayload
from .scheduler import HeapScheduler, Scheduler
from .state import PlatformState
from .system import RFSystem, TypeCheckError

__version__ = "0.1.0"

__all__ = [
    "Component",
    "MergeComponent",
    "delays",
    "PlatformState",
    "Event",
    "DataObject",
    "SignalPayload",
    "PulseBuffer",
    "Spectrogram",
    "DetectionReport",
    "SIGNAL_RX",
    "DEFAULT_IQ_DTYPE",
    "Scheduler",
    "HeapScheduler",
    "RFSystem",
    "TypeCheckError",
]
