# Modular Monolith vs. Microservices — a minimum viable comparison

A minimal, working C++ example that implements **the same signal-processing
pipeline twice**, once as a modular monolith and once as two microservices,
so the architectural trade-off is visible in code rather than in the
abstract.

The pipeline: detect pulses in a stream of IQ (in-phase/quadrature) samples,
then compute summary statistics over the pulses found (count, mean peak
amplitude, mean duration, mean pulse-repetition interval). This is a toy
version of a real signal-intelligence front end, small enough to read start
to finish.

## The point of the example

Both variants are built from the **exact same business logic** and the
**exact same protobuf message schema** (`proto/pulse.proto`). Only the
boundary between the "detector" module and the "stats" module changes:

| | Modular monolith (`monolith_app`) | Microservices (`detector_service` + `stats_service`) |
|---|---|---|
| Process model | 1 process | 2 processes |
| Module loading | `dlopen()` at startup | separate binaries, started independently |
| Module boundary | C ABI function call (`module_api.h`) | TCP socket |
| Data crossing the boundary | serialized `pulse.PulseEventBatch` bytes, passed as a pointer + length | serialized `pulse.PulseEventBatch` bytes, length-prefixed on the wire |
| Failure domain | one module crashing takes down the process | one service crashing is isolated, visible as a dropped connection |
| Deployment | ship one executable + two `.so` files, versioned together | ship/scale/deploy each service independently |

The `pulse.PulseEventBatch` protobuf message is what proves the point: it's
the same bytes either way. Swapping `dlopen()` + a function call for
`connect()` + `send()` is the *entire* difference between the two
architectures here — the detection and statistics algorithms
(`common/src/pulse_detector.cpp`, `common/src/pulse_stats.cpp`) don't know
or care which one is in use.

## Layout

```
proto/pulse.proto          Shared message schema (IQSample, PulseEvent, PulseSummary, ...)

common/                    Architecture-agnostic core ("pulsecore")
  include/pulse_detector.h   Amplitude-threshold pulse detector
  include/pulse_stats.h      Running summary-statistics accumulator
  include/iq_source.h        Deterministic synthetic IQ generator (same input, both builds)
  src/...

monolith/                  Modular monolith build
  include/module_api.h       C ABI every plugin module exports (create/process/destroy)
  plugins/
    pulse_detector_plugin.cpp  -> libpulse_detector_plugin.so
    pulse_stats_plugin.cpp     -> libpulse_stats_plugin.so
  host/monolith_main.cpp     Single executable; dlopen()s both modules, wires them together

microservice/               Two-executable build
  net/framing.{h,cpp}         Minimal length-prefixed protobuf-over-TCP framing
  detector_service/main.cpp   Executable #1: detects pulses, streams PulseEventBatch over TCP
  stats_service/main.cpp      Executable #2: TCP server, accumulates PulseEventBatch, prints summary

scripts/run_monolith.sh       Build + run the monolith
scripts/run_microservices.sh  Build + run both microservices
```

## Building

Requires CMake, a C++17 compiler, and protobuf (`protobuf-compiler` +
`libprotobuf-dev` on Debian/Ubuntu).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces, all in `build/bin/`:

- `monolith_app` — the modular monolith host
- `libpulse_detector_plugin.so`, `libpulse_stats_plugin.so` — its dynamically loaded modules
- `detector_service`, `stats_service` — the microservice pair
- `libpulse_proto.so` — the generated protobuf message code, shared by everything (see note below)

## Running

**Modular monolith** (one process, two `dlopen()`'d modules):

```sh
./scripts/run_monolith.sh
# or directly:
./build/bin/monolith_app ./build/bin
```

**Microservices** (two processes, protobuf over TCP):

```sh
./scripts/run_microservices.sh
# or directly, in two terminals:
./build/bin/stats_service 50051
./build/bin/detector_service 127.0.0.1 50051
```

Both print the same detected pulse count and summary statistics, computed
from the same synthetic IQ data (six rectangular pulses buried in noise) —
that's the point: it's the same result, produced two structurally different
ways.

## A gotcha this example ran into (and why the build looks the way it does)

protobuf-generated message code registers its message descriptors in a
process-global registry the first time it's loaded. `monolith_app` links
the detector/stats business logic directly *and* `dlopen()`s two plugin
modules that link the same logic — if the generated `pulse.pb.cc` code were
statically linked into each of those three places, all three would try to
register `pulse.proto` in the same process and it would abort with a
duplicate-registration `CHECK` failure at startup.

The fix (see `CMakeLists.txt`): the generated protobuf code lives in its
own shared library, `pulse_proto`, that everything links *dynamically*.
The dynamic linker guarantees a given `.so` is loaded into a process at
most once, so the descriptors get registered exactly once no matter how
many modules depend on them. The rest of the shared logic (`pulsecore`,
the detector/stats/IQ-source code) has no such global state, so it's fine
for that to be statically linked into multiple binaries in the same
process — only generated protobuf code needs this treatment.

This is a real constraint of building modular-monolith-style plugin systems
in C++ with protobuf, not an artifact of the demo, so it's left in as-is
rather than hidden.
