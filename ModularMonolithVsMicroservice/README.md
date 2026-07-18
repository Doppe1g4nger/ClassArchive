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
| Module boundary | typed C++ reference (`module_api.h`) | TCP socket |
| Data crossing the boundary | a `pulse::IQBatch&` / `pulse::PulseEventBatch&`, in place, no copy | serialized `pulse.PulseEventBatch` bytes, length-prefixed on the wire |
| Failure domain | one module crashing takes down the process | one service crashing is isolated, visible as a dropped connection |
| Deployment | ship one executable + two `.so` files, versioned together | ship/scale/deploy each service independently |

The `pulse.*` protobuf schema is what proves the point: it's the same
message *types* either way, defined once in `proto/pulse.proto` and used by
every module. What differs is what those types have to go through to cross
each architecture's module boundary. `monolith_app`'s modules run in the
same process and, as of this build, share one loaded copy of the generated
protobuf code (see the shared `pulse_proto` library below), so they pass
`pulse::IQBatch`/`pulse::PulseEventBatch` objects by reference — no
serialize, no parse. `detector_service` and `stats_service` are separate
processes with separate address spaces; there is no way to hand one a
pointer into the other's memory, so they *must* serialize onto the socket.
That's not an implementation gap in this demo, it's the actual, unavoidable
cost of a real process boundary — which is exactly the trade-off this
example exists to make visible.

## Layout

```
proto/pulse.proto          Shared message schema (IQSample, PulseEvent, PulseSummary, ...)

common/                    Architecture-agnostic core ("pulsecore")
  include/pulse_detector.h   Amplitude-threshold pulse detector
  include/pulse_stats.h      Running summary-statistics accumulator
  include/iq_source.h        Deterministic synthetic IQ generator (same input, both builds)
  src/...

monolith/                  Modular monolith build
  include/module_api.h       Typed dlsym()-able ABI every plugin module exports (create/process/destroy)
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

That fix turned out to pay for itself twice over. Having exactly one
loaded definition of `pulse::IQBatch` and friends isn't just what avoids
the duplicate-registration crash — it's also the precondition for what
`module_api.h` does now: pass `pulse::` messages across the `dlopen()`
boundary by reference instead of by serialized bytes (see "Eliminating the
monolith's serialization tax" below). Without a single shared
`pulse_proto`, each plugin would carry its own copy of those types with
its own vtable layout, and passing a reference across the boundary would
be undefined behavior instead of just safe.

## Optimizations

### Eliminating the monolith's serialization tax

The first pass at this repo made `module_api.h` byte-oriented on purpose —
modules took `const uint8_t*`/length and returned a heap-allocated
`uint8_t*`/length, mirroring the microservices' wire format so the same
`PulseEventBatch` bytes visibly crossed both kinds of boundary. That
looked symmetric, but it forced `monolith_app` to `SerializeToString()`
every `IQBatch` (the largest message in the pipeline, 4096 samples/batch)
just to hand it to the detector plugin, which immediately `ParseFromArray`'d
it back — a real cost paid for a boundary that doesn't need it, since host
and plugin share one address space.

The fix: `module_api.h`'s function-pointer types now take/return
`pulse::IQBatch&`, `pulse::PulseEventBatch&`, and `pulse::PulseSummary*`
directly (see the header for the full rationale). `monolith_main.cpp` no
longer serializes anything in its per-batch loop; `pulse_detector_plugin.cpp`
and `pulse_stats_plugin.cpp` no longer parse or allocate an output buffer.
This only works safely because of the `pulse_proto` shared-library fix
above — every module has the *same* definition of every `pulse::` type, so
passing them by reference across `dlopen()` is well-defined. The
microservices can't take this shortcut: `detector_service` and
`stats_service` are separate processes, so `PulseEventBatch` genuinely has
to be serialized to cross that boundary. That asymmetry — not an
implementation gap — is now the actual difference the benchmark below
measures.

### Everything else

Both architectures also share a few more ordinary tuning passes:

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

Representative result (this machine, current build — modules pass typed
references, no serialization inside `monolith_app`):

```
Results over 50 runs, 1000 pulses/run:

            modular monolith  min=10.8ms  median=11.7ms  mean=12.1ms  stdev=1.4ms  max=17.0ms
               microservices  min=18.5ms  median=20.1ms  mean=20.1ms  stdev=0.8ms  max=22.8ms

microservices mean is ~1.7x the modular monolith mean
```

This matches intuition now: the monolith, doing real work through plain
function calls with no serialization, is roughly 40% faster than two
processes serializing `PulseEventBatch` over a loopback TCP socket.

That gap is even more pronounced when there's barely any work to amortize
process-startup cost against — run `./scripts/benchmark.sh 20 0` (0
pulses, i.e. one nearly-empty batch) and the monolith wins by roughly 2x
(~4.8ms vs ~9.8ms), since a single process start beats two process starts
plus a TCP handshake regardless of how the modules talk once they're up.

**This wasn't always the result.** An earlier version of `module_api.h`
made the monolith's modules speak bytes instead of typed references, on
the theory that it would make the two architectures more directly
comparable — same `PulseEventBatch` bytes crossing both kinds of boundary.
At the same 1000-pulse workload, that version's monolith actually lost to
the microservices (mean ~26.7ms vs ~19.6ms, i.e. microservices ~25%
*faster*), because forcing the monolith to serialize/deserialize every
`IQBatch` just to hand it across an in-process `dlopen()` boundary cost
more than the microservices' extra process and socket overhead did at that
batch size. Passing typed references instead (see "Eliminating the
monolith's serialization tax" above) removed that unnecessary cost and
restored the expected ordering — proof that the earlier byte-oriented
design was measuring an artifact of its own ABI choice, not something
inherent to modular monoliths.

### Correctness

The optimizations don't change output: both binaries were re-run at
6 pulses (the original default) and 1000 pulses after every change and
produced byte-identical summary statistics (`pulse_count`,
`mean_peak_amplitude`, `mean_duration_seconds`, `mean_pri_seconds`,
`min_peak_amplitude`, `max_peak_amplitude`) between the monolith and
microservice builds in every case.
