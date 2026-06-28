"""Demo: platform state (6DOF + name) and transmitting back to the environment.

The RFSystem models a platform with a name and a 6DOF state. A Transmitter sends
a signal buffer back to the environment via the system's ``on_transmit`` hook
(the egress counterpart of ``signalRX``); each transmission carries a snapshot of
the platform's 6DOF state so the environment can model propagation.

Shows both transmitter modes:
  * a Repeater (relay): an incoming signalRX is amplified and retransmitted,
  * a ToneTransmitter (source): emits a CW tone when fired.

Run with::

    python examples/demo_transmit.py
"""

from __future__ import annotations

import numpy as np

from rfdes import HeapScheduler, PlatformState, RFSystem
from rfdes.components import Repeater, ToneTransmitter


def main() -> None:
    sched = HeapScheduler()
    egress_log = []

    def environment_on_transmit(payload, state):
        """Stand-in environment: receives the platform's transmission."""
        egress_log.append((sched.now(), payload, state))
        print(f"  [t={sched.now()*1e9:6.1f} ns] environment received TX from "
              f"{state.name!r} at position {state.position.tolist()} "
              f"(vel {state.velocity.tolist()}): {payload.num_samples} samples, "
              f"fc={payload.center_freq/1e9:.3f} GHz")

    state = PlatformState(
        name="Platform-A",
        position=[1000.0, 2000.0, 500.0],
        velocity=[-50.0, 0.0, 0.0],
        orientation=[1.0, 0.0, 0.0, 0.0],   # identity quaternion
    )
    system = RFSystem(sched, name="Platform-A", state=state,
                      on_transmit=environment_on_transmit)

    # Relay: signalRX -> amplify -> retransmit.
    repeater = system.add(Repeater("repeater", gain_db=20.0, processing_delay=5e-9))
    system.set_entry(repeater)

    # Source: a beacon tone transmitter, fired on demand.
    beacon = system.add(ToneTransmitter("beacon", freq=1e6, sample_rate=10e6,
                                        num_samples=256, center_freq=2.4e9,
                                        processing_delay=2e-9))

    print(f"Platform {system.name!r} state: pos={state.position.tolist()}, "
          f"quat={state.orientation.tolist()}\n")

    print("Repeater relays an incoming signalRX:")
    fs = 10e6
    iq = (0.1 * np.exp(2j * np.pi * 1e6 * np.arange(512) / fs)).astype(np.complex64)
    system.on_signal_rx(iq, sample_rate=fs, center_freq=1.5e9)

    print("Beacon transmits a tone (source mode); platform has since moved:")
    system.state.position = np.array([800.0, 2000.0, 500.0])  # platform moved
    beacon.fire()

    sched.run()
    print(f"\ntotal transmissions to environment: {len(egress_log)}")


if __name__ == "__main__":
    main()
