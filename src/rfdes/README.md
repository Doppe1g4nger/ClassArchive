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
  event. By default the buffer is delivered immediately (`delay=0`), interleaving
  correctly with the environment's other same-timestamp events; pass
  `at=<scheduler time>` to schedule the delivery at an arbitrary future time on
  the queue, like any normal event (the sample `start_time` defaults to that
  delivery time).

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

## Closed-loop feedback (control port)

A `ControllableComponent` exposes a reserved `"control"` input port: a
`ControlMessage` delivered there is handled by `on_control(msg)` (which mutates
state) and produces no downstream output. This lets a downstream component
reconfigure an upstream one — a feedback cycle — without a runaway loop, since the
back-edge carries control rather than signal. Wire it with the usual port refs:

```python
from rfdes.components import TunableBandpassFilter, PulseDetector, ScanScheduler, Recorder

filt >> det
det  >> rec
det  >> scan
scan >> filt["control"]   # feedback: retune the front-end filter
```

`ScanScheduler` watches the detector and retunes `TunableBandpassFilter` band by
band until pulses appear, then dwells — see `examples/demo_feedback.py`.

Wire merges with `>>` by targeting a named port via `component[port]`:

```python
lna >> split
split >> pulse_det
split >> spectro
pulse_det >> fusion["pulses"]        # -> the merge's "pulses" port
spectro   >> fusion["spectrogram"]   # -> the merge's "spectrogram" port
fusion >> recorder
```

## Dynamic (data-dependent) delays

`processing_delay` may be a constant **or a callable**. The callable is invoked
with the component's *input* (for a merge, the matched `{port: data}` dict) and
returns the delay for that firing — so latency can depend on the data or be
random. Ready-made factories live in `rfdes.delays`:

```python
from rfdes.delays import per_sample, per_pulse, jitter

Amplifier("lna", gain_db=20, processing_delay=jitter(5e-9, 1e-9))           # random
PulseDetector("pd", threshold=0.5, processing_delay=per_sample(1e-9, 1e-11)) # ~ IQ length
PulseRelay("relay", processing_delay=per_pulse(1e-9, 5e-10))                 # ~ #pulses
```

Any `callable(input) -> float` works; the delay is resolved once per firing
(so fan-out branches share one consistent value) and must be non-negative.

## Blocking / queueing while processing

By default a component is non-blocking (it can process overlapping inputs). Pass
`when_busy` to model a component that processes one item at a time and is busy for
its `processing_delay`:

```python
Component("slow", processing_delay=3.0, when_busy="queue")  # late inputs wait (FIFO)
Component("slow", processing_delay=3.0, when_busy="drop")   # late inputs discarded
```

The component exposes `processing` (True while busy) and `dropped` (count under
`"drop"`). Queued items incur real queueing delay. Applies to single-input
components (and `Transmitter` relay mode); `MergeComponent` stays non-blocking.

## Platform state and transmitting back to the environment

An `RFSystem` models a platform: it carries a `name` and a 6DOF `PlatformState`
(position, velocity, quaternion `orientation`, `angular_velocity`, `metadata`).
Components can read it via `self.system.state`.

A `Transmitter` sends a buffer back to the environment through a settable
`on_transmit(payload, state)` hook (the egress counterpart of `signalRX`); the
system passes a snapshot of the platform's 6DOF state with each transmission.

```python
from rfdes import RFSystem, PlatformState, HeapScheduler
from rfdes.components import Repeater, ToneTransmitter

def on_transmit(payload, state):       # supplied by the environment integration
    ...                                 # state.position / .velocity / .orientation

state = PlatformState(name="A", position=[1000, 2000, 500], velocity=[-50, 0, 0])
system = RFSystem(HeapScheduler(), name="A", state=state, on_transmit=on_transmit)

rep = system.add(Repeater("rep", gain_db=20))      # relay: signalRX -> retransmit
system.set_entry(rep)
beacon = system.add(ToneTransmitter("beacon", freq=1e6, sample_rate=10e6, num_samples=256))
beacon.fire()                                       # source: emit a tone
```

## Examples & tests

```bash
pip install -e ".[dev]"
pytest
python examples/demo_chain.py
python examples/demo_fanout.py
python examples/demo_multitype.py          # multiple data types + validation + merge
python examples/demo_split_merge_rshift.py # split + merge wired with >>
python examples/demo_dynamic_delay.py      # data-dependent and random delays
python examples/demo_transmit.py           # platform 6DOF state + transmit egress
python examples/demo_blocking.py           # blocking: queue vs drop while busy
python examples/demo_jammer.py             # capstone: detect pulses, jam @ 2.4 GHz
python examples/demo_feedback.py           # closed loop: scan scheduler retunes filter
```

The capstone `demo_jammer.py` ties everything together: an EW platform (with 6DOF
state) ingests `signalRX`, splits to a `PulseDetector` and `Spectrogrammer`,
fuses them (`DetectionFusion`), and a `JamController` emits a barrage-noise jam
back to the environment — but only when pulses are present **and** the carrier is
at 2.4 GHz. The jammer `Transmitter` uses `when_busy="drop"`, so a rapid burst
shows jams being dropped while it is busy.
