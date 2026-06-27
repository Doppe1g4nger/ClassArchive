# rfdes — component-level discrete-event RF system framework

`rfdes` lets you model an entire RF system at the component level inside a
discrete-event simulation. It is designed to run **inside** an external
discrete-event simulator that already models the RF *environment* and owns the
master clock and event queue. That environment hands the RF system a buffer of
IQ samples through a special `signalRX` event; `rfdes` then models each
component's execution and the data flow between components as further events
placed onto the same queue.

## Concepts

- **Scheduler (the integration boundary).** `rfdes` never owns time. It talks to
  the host event queue through the small `Scheduler` Protocol
  (`now()`, `schedule(delay, callback)`, `cancel(handle)`). Adapt it to any DES
  engine (SimPy, a custom queue, asyncio). A standalone `HeapScheduler` is
  included so you can run and test systems on their own.
- **Component.** A node in the system graph. Override `on_signal(payload)` to
  model behavior (gain, mixing, filtering, ...). Wire components with
  `subscribe` (or `>>`). Fan-out is just multiple subscribers.
- **Per-component delay.** Each component has a `processing_delay`. It is charged
  *on emit*: a component's output reaches its subscribers `processing_delay`
  later on the queue, modeling that component's latency.
- **SignalPayload.** An immutable buffer of IQ samples (`numpy` complex array)
  plus `sample_rate`, `center_freq`, `start_time`, and a `metadata` dict.
  Transforms return a *new* payload, so fan-out never aliases.
- **RFSystem.** Holds components, binds them to the scheduler, and exposes
  `on_signal_rx(...)` — the method the external simulator calls on a `signalRX`
  event. The handoff is scheduled at `delay=0` so it interleaves correctly with
  the environment's other same-timestamp events instead of jumping ahead.

## Quick start

```python
import numpy as np
from rfdes import RFSystem, HeapScheduler
from rfdes.components import Amplifier, Mixer, ADC, Recorder

sched = HeapScheduler()
system = RFSystem(sched)

lna = system.add(Amplifier("LNA", gain_db=20, processing_delay=1e-9))
mix = system.add(Mixer("mix", lo_freq=1e9, processing_delay=2e-9))
adc = system.add(ADC("ADC", bits=12, full_scale=20, processing_delay=5e-9))
rec = system.add(Recorder("baseband"))

lna >> mix >> adc >> rec          # wire the chain
system.set_entry(lna)             # who receives signalRX

# Stand in for the RF environment: deliver an IQ buffer.
fs = 10e6
iq = (0.1 * np.exp(2j*np.pi*2e6*np.arange(1024)/fs)).astype(np.complex64)
system.on_signal_rx(iq, sample_rate=fs, center_freq=1.5e9, t=100e-9)

sched.run()
print(rec.payloads[-1].center_freq, rec.payloads[-1].metadata["gain_db"])
```

## Plugging into your external simulator

Implement the `Scheduler` Protocol as a thin adapter over the host queue, then
construct `RFSystem(adapter)` and call `system.on_signal_rx(...)` from the host's
`signalRX` handler:

```python
class HostSchedulerAdapter:
    def __init__(self, host): self.host = host
    def now(self): return self.host.current_time()
    def schedule(self, delay, callback):
        return self.host.enqueue(at=self.host.current_time() + delay, fn=callback)
    def cancel(self, handle): self.host.dequeue(handle)
```

Every component-processing event and inter-component delivery then lands on the
host's event queue, interleaved with the RF environment's own events.

## Build your own component

```python
from dataclasses import replace
from rfdes import Component

class Gain(Component):
    def __init__(self, name, factor, processing_delay=0.0):
        super().__init__(name, processing_delay)
        self.factor = factor
    def on_signal(self, payload):
        return replace(payload, iq=(payload.iq * self.factor).astype(payload.iq.dtype))
```

Return `None` from `on_signal` to make a terminal sink.

## Multiple data types, validation, and merging

Components can exchange **different data types**, not just IQ. Every concrete
payload subclasses `DataObject` (e.g. `SignalPayload`, `PulseBuffer`,
`Spectrogram`, `DetectionReport`). A component declares the types it `accepts`
and the type it `produces`:

```python
from rfdes import Component
from rfdes.datatypes import PulseBuffer

class PulseDetector(Component):
    accepts = (SignalPayload,)
    produces = PulseBuffer
    def on_signal(self, payload):
        ...  # return a PulseBuffer
```

**Pre-simulation type check.** `RFSystem.validate()` walks every connection and
raises a `TypeCheckError` (listing *all* problems) if a producer's `produces`
type isn't accepted at the downstream port, if the entry can't accept
`SignalPayload`, or if a merge port is left unfed. It runs automatically on the
first `signalRX`, and you can call it explicitly.

**Merge / join.** Subclass `MergeComponent` to fire only once *every* input port
has data. Declare named ports in `inputs`, wire producers to them with
`subscribe(merge, port="...")`, and implement `on_merge(inputs)`:

```python
from rfdes import MergeComponent
from rfdes.datatypes import PulseBuffer, Spectrogram, DetectionReport

class DetectionFusion(MergeComponent):
    inputs = {"pulses": (PulseBuffer,), "spectrogram": (Spectrogram,)}
    produces = DetectionReport
    def on_merge(self, inputs):
        ...  # combine inputs["pulses"] and inputs["spectrogram"]

pulse_det.subscribe(fusion, port="pulses")
spectro.subscribe(fusion, port="spectrogram")
```

By default a merge does a FIFO *zip* (one item per port). Override
`correlation_key(data)` to instead match items that share a key.

## Examples & tests

```bash
pip install -e ".[dev]"
pytest
python examples/demo_chain.py
python examples/demo_fanout.py
python examples/demo_multitype.py   # multiple data types + validation + merge
```
