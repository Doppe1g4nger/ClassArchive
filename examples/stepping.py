"""Helpers for stepping through an rfdes simulation one event at a time.

These let a demo recover the current simulation time from the scheduler at every
event and pause between events. The scheduler stores opaque callbacks, so to
report *which* component fired we instrument each component's ``on_signal`` to
append a record to a shared log; one ``scheduler.step()`` fires exactly one
component, so ``log[-1]`` is always the event that just fired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rfdes import Component, SignalPayload
from rfdes.scheduler import Scheduler


@dataclass
class FireRecord:
    """One component firing: its name and the payload it took in / put out."""

    name: str
    inp: SignalPayload
    out: Optional[SignalPayload]


def instrument(component: Component, log: list[FireRecord]) -> Component:
    """Wrap ``component.on_signal`` so every firing is appended to ``log``.

    Returns the component for convenience. The override is an instance attribute
    that shadows the class method; ``Component.receive`` looks up ``on_signal``
    at call time, so the wrapper is picked up when the scheduler fires the event.
    """
    original = component.on_signal  # bound method captured before override

    def logged(payload: SignalPayload) -> Optional[SignalPayload]:
        out = original(payload)
        log.append(FireRecord(name=component.name, inp=payload, out=out))
        return out

    component.on_signal = logged  # type: ignore[method-assign]
    return component


def summarize(payload: Optional[SignalPayload]) -> str:
    """One-line summary of a component's output payload."""
    if payload is None:
        return "absorbed (sink)"
    parts = [f"fc={payload.center_freq / 1e9:.3f} GHz", f"N={payload.num_samples}"]
    if "gain_db" in payload.metadata:
        parts.append(f"gain={payload.metadata['gain_db']:.1f} dB")
    return ", ".join(parts)


def step_through(sched: Scheduler, log: list[FireRecord]) -> int:
    """Fire events one at a time, querying the clock and pausing at each step.

    Pauses for the user to press Enter between events. On EOF (piped / non-
    interactive input) it drains the rest of the queue and returns. Returns the
    number of events fired.
    """
    step = 0
    while sched.step():
        step += 1
        rec = log[-1]
        # Simulation time is recovered from the scheduler at each event.
        print(f"[step {step}] t = {sched.now() * 1e9:8.2f} ns  "
              f"{rec.name:<10} -> {summarize(rec.out)}")
        try:
            input("    press Enter for next event (Ctrl-D to finish) ...")
        except EOFError:
            print("\n    (no more input; draining remaining events)")
            # Drain whatever is left without further pauses.
            while sched.step():
                step += 1
                rec = log[-1]
                print(f"[step {step}] t = {sched.now() * 1e9:8.2f} ns  "
                      f"{rec.name:<10} -> {summarize(rec.out)}")
            break
    return step
