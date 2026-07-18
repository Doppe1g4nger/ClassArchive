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
scripts/benchmark.sh          Build + time N runs of each architecture
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
./build/bin/monolith_app <plugin_dir> [num_pulses]
./build/bin/monolith_app ./build/bin 1000
```

**Microservices** (two processes, protobuf over TCP):

```sh
./scripts/run_microservices.sh
# or directly, in two terminals:
./build/bin/stats_service <port>
./build/bin/detector_service <host> <port> [num_pulses]

./build/bin/stats_service 50051
./build/bin/detector_service 127.0.0.1 50051 1000
```

Both print the same detected pulse count and summary statistics, computed
from the same synthetic IQ data (`num_pulses` rectangular pulses buried in
noise, 6 by default) — that's the point: it's the same result, produced two
structurally different ways.

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

## Optimizations

Both architectures share the same handful of obvious tuning passes:

- **Bigger batches.** `SyntheticIQSource`'s batch size went from 256 to
  4096 samples (`common/include/iq_source.h`). Every batch costs one fixed
  overhead — a protobuf serialize/parse, one module call or one socket
  message — so processing the same amount of data in fewer, larger batches
  amortizes that fixed cost over more work. This helps the monolith some
  and the microservices a lot, since a socket round trip's fixed cost
  (syscalls, kernel copies) is much larger than an in-process call's.
- **Reused protobuf message objects.** Every hot loop (`monolith_main.cpp`,
  both plugins, both microservice `main.cpp`s) used to declare a fresh
  `pulse::IQBatch` / `PulseEventBatch` as a local inside the loop body.
  Messages with repeated fields keep heap-allocated backing arrays;
  constructing a new message every iteration means allocating and freeing
  those arrays every iteration too. Hoisting the messages to loop scope
  (or a module's member state) and letting `Parse*()`'s implicit `Clear()`
  reset them in place turns that into "allocate once, reuse after."
- **Reused serialization buffers.** Same idea for the `std::string` buffers
  fed to `SerializeToString`: declared once outside the loop instead of
  per-iteration, so their capacity carries over instead of being
  reallocated from scratch every batch.
- **`TCP_NODELAY`** on both ends of the microservice socket
  (`microservice/net/framing.cpp`), disabling Nagle's algorithm so small
  writes aren't held back waiting to coalesce with more outgoing data.
- **One `writev()` instead of two `send()`s** for each framed message —
  the 4-byte length prefix and the payload go out in a single syscall via
  a 2-element `iovec` instead of two separate `send()` calls.
- `-O3 -DNDEBUG` was already CMake's default for `CMAKE_BUILD_TYPE=Release`
  (verified with `--verbose`), so there was nothing to change there.

None of this changes behavior — see "Correctness" below.

## Benchmark: 50 runs, 1000 pulses

`scripts/benchmark.sh` builds both architectures, then times 50
independent, fresh-process runs of each against a synthetic 1000-pulse
input (`./scripts/benchmark.sh 50 1000`). Each run is timed end to end:
process start to process exit for `monolith_app`; for the microservice
pair, from *before* `stats_service` is even launched to after both
processes have exited, so process-startup cost is counted symmetrically
on both sides.

Representative result (this machine, three separate 50-run measurements
all landed in the same range):

```
Results over 50 runs, 1000 pulses/run:

            modular monolith  min=24.5ms  median=25.6ms  mean=26.7ms  stdev=4.7ms  max=50.9ms
               microservices  min=18.1ms  median=19.3ms  mean=19.6ms  stdev=1.0ms  max=23.3ms

microservices mean is ~0.75x the modular monolith mean
```

**The microservices come out faster here, which is the opposite of the
naive intuition** ("function call must beat a socket round trip"). It's
real and reproducible, and the reason is a genuine architectural insight,
not a benchmarking artifact:

`module_api.h` deliberately makes the monolith's plugin boundary
byte-oriented — the same wire format the microservices use — so the demo
can show the same `PulseEventBatch` bytes crossing both kinds of boundary.
That means `monolith_app` pays to **serialize the raw `IQBatch` and
deserialize it again** just to hand it across the `dlopen()` boundary into
the detector plugin, even though host and plugin share one address space
and, in a less symmetry-obsessed design, could just pass a pointer.
`IQBatch` (4096 samples/batch) is the largest message in this pipeline by
far — much bigger than the `PulseEventBatch` it produces (a handful of
detected pulses per batch). `detector_service`, by contrast, calls
`PulseDetector::Process()` directly on the in-memory `IQBatch` with no
serialization at all, and only serializes the much smaller
`PulseEventBatch` for the network hop. So per batch, the monolith does
strictly more (de)serialization work than the microservices do, and at
1000 pulses that cost outweighs the microservices' extra process and
socket overhead.

That overhead is real too, and dominates at small workloads: run the same
benchmark with `num_pulses=0` (`./scripts/benchmark.sh 20 0`) and the
monolith wins by roughly 2x (~5.5ms vs ~10.5ms), because there's almost no
detection work to amortize the cost of starting a second process and
setting up a TCP connection. The crossover — where the monolith's
per-batch serialization tax overtakes the microservices' fixed process/IPC
tax — happens somewhere between those two workloads.

The honest takeaway: this particular monolith's plugin ABI accepted a
real serialization cost in exchange for looking structurally identical to
the microservice wire boundary, which is what makes the two architectures
directly comparable in the first place. A production modular monolith
that let modules share C++ types instead of only bytes wouldn't pay that
tax and would very likely win the 1000-pulse case — but it would also lose
the property that makes this repo's comparison legible: that the *same
bytes* cross both kinds of boundary.

### Correctness

The optimizations don't change output: both binaries were re-run at
6 pulses (the original default) and 1000 pulses after every change and
produced byte-identical summary statistics (`pulse_count`,
`mean_peak_amplitude`, `mean_duration_seconds`, `mean_pri_seconds`,
`min_peak_amplitude`, `max_peak_amplitude`) between the monolith and
microservice builds in every case.
