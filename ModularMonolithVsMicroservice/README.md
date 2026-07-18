# Modular Monolith vs. Microservices — a minimum viable comparison

A minimal, working C++ example that implements **the same signal-processing
pipeline twice**, once as a modular monolith and once as a set of
microservices, so the architectural trade-off is visible in code rather than
in the abstract.

The pipeline is a toy signal-intelligence front end with five stages, all
fed from one synthetic IQ (in-phase/quadrature) sample stream:

- **detector** — amplitude-threshold pulse detection
- **stats** — running summary statistics over detected pulses (count, mean
  peak amplitude, mean duration, mean pulse-repetition interval)
- **spectrogram** — a small-bin magnitude spectrum estimate per batch
- **jammer** — flags batches with sustained high-power duty cycle
- **deinterleaver** — groups pulses into candidate emitter tracks by PRI

`spectrogram` and `jammer` consume the same raw IQ batches the detector
does; `stats` and `deinterleaver` both consume the detector's derived pulse
events. Small enough to read start to finish, but with enough independent
stages to show what "add a module" costs in each architecture.

## The point of the example

Both variants are built from the **exact same business logic** and the
**exact same protobuf message schema** (`proto/pulse.proto`). Only the
boundary between modules changes:

| | Modular monolith (`monolith_app`) | Microservices (`detector_service` + 4 consumers) |
|---|---|---|
| Process model | 1 process | 5 processes |
| Module loading | `dlopen()` at startup | separate binaries, started independently |
| Module boundary | typed C++ reference (`module_api.h`) | TCP socket |
| Data crossing the boundary | a `pulse::IQBatch&` / `pulse::PulseEventBatch&`, in place, no copy | serialized `pulse.*` bytes, length-prefixed on the wire |
| Failure domain | one module crashing takes down the process | one service crashing is isolated, visible as a dropped connection |
| Deployment | ship one executable + five `.so` files, versioned together | ship/scale/deploy each service independently |

The `pulse.*` protobuf schema is what proves the point: it's the same
message *types* either way, defined once in `proto/pulse.proto` and used by
every module. What differs is what those types have to go through to cross
each architecture's module boundary. `monolith_app`'s modules run in the
same process and, as of this build, share one loaded copy of the generated
protobuf code (see the shared `pulse_proto` library below), so they pass
`pulse::IQBatch`/`pulse::PulseEventBatch` objects by reference — no
serialize, no parse. `detector_service` and its four downstream consumers
are separate processes with separate address spaces; there is no way to
hand one a pointer into another's memory, so they *must* serialize onto
the socket. That's not an implementation gap in this demo, it's the
actual, unavoidable cost of a real process boundary — which is exactly the
trade-off this example exists to make visible.

`detector_service` is the microservice build's data-plane hub: it
generates the synthetic IQ stream, runs detection locally (same as the
monolith's detector module), and fans both the raw batches and its own
derived pulse events out to `stats_service`, `spectrogram_service`,
`jammer_service`, and `deinterleave_service` — one TCP connection per
consumer, since a real process boundary has no broadcast primitive of its
own. `monolith_app` does the equivalent fan-out with five ordinary function
calls in a loop.

## Layout

```
proto/pulse.proto          Shared message schema (IQSample, PulseEvent, PulseSummary,
                            SpectrogramSummary, JamSummary, EmitterTrack/DeinterleaveSummary, ...)

common/                    Architecture-agnostic core ("pulsecore")
  include/pulse_detector.h   Amplitude-threshold pulse detector
  include/pulse_stats.h      Running summary-statistics accumulator
  include/spectrogram.h      Small-bin magnitude spectrum estimator
  include/jammer.h           Duty-cycle-based jamming detector
  include/deinterleaver.h    PRI-based emitter track grouping
  include/iq_source.h        Deterministic synthetic IQ generator (same input, both builds)
  src/...

monolith/                  Modular monolith build
  include/module_api.h       Typed dlsym()-able ABI every plugin module exports (create/process/destroy)
  plugins/
    pulse_detector_plugin.cpp      -> libpulse_detector_plugin.so
    pulse_stats_plugin.cpp         -> libpulse_stats_plugin.so
    pulse_spectrogram_plugin.cpp   -> libpulse_spectrogram_plugin.so
    pulse_jammer_plugin.cpp        -> libpulse_jammer_plugin.so
    pulse_deinterleaver_plugin.cpp -> libpulse_deinterleaver_plugin.so
  host/monolith_main.cpp     Single executable; dlopen()s all five modules, wires them together

microservice/               Five-executable build
  net/framing.{h,cpp}          Minimal length-prefixed protobuf-over-TCP framing
  detector_service/main.cpp    Data-plane hub: detects pulses, fans raw batches + events out to the other four
  stats_service/main.cpp       TCP server, accumulates PulseEventBatch, prints summary
  spectrogram_service/main.cpp TCP server, accumulates IQBatch into a running spectrum
  jammer_service/main.cpp      TCP server, accumulates IQBatch into running jam-detection state
  deinterleave_service/main.cpp TCP server, accumulates PulseEventBatch into emitter tracks

scripts/run_monolith.sh       Build + run the monolith
scripts/run_microservices.sh  Build + run all five microservices
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
- `libpulse_detector_plugin.so`, `libpulse_stats_plugin.so`,
  `libpulse_spectrogram_plugin.so`, `libpulse_jammer_plugin.so`,
  `libpulse_deinterleaver_plugin.so` — its dynamically loaded modules
- `detector_service`, `stats_service`, `spectrogram_service`,
  `jammer_service`, `deinterleave_service` — the microservice set
- `libpulse_proto.so` — the generated protobuf message code, shared by everything (see note below)

## Running

**Modular monolith** (one process, five `dlopen()`'d modules):

```sh
./scripts/run_monolith.sh
# or directly:
./build/bin/monolith_app <plugin_dir> [num_pulses]
./build/bin/monolith_app ./build/bin 1000
```

**Microservices** (five processes, protobuf over TCP). `detector_service`
connects to the other four at `base_port + offset`: `stats_service` at
`+0`, `spectrogram_service` at `+1`, `jammer_service` at `+2`,
`deinterleave_service` at `+3`.

```sh
./scripts/run_microservices.sh
# or directly, in five terminals:
./build/bin/stats_service <base_port>
./build/bin/spectrogram_service <base_port + 1>
./build/bin/jammer_service <base_port + 2>
./build/bin/deinterleave_service <base_port + 3>
./build/bin/detector_service <host> <base_port> [num_pulses]

./build/bin/stats_service 50051
./build/bin/spectrogram_service 50052
./build/bin/jammer_service 50053
./build/bin/deinterleave_service 50054
./build/bin/detector_service 127.0.0.1 50051 1000
```

Both print the same detected pulse count, summary statistics, spectrum,
jam-detection state, and emitter tracks, computed from the same synthetic
IQ data (`num_pulses` rectangular pulses buried in noise, 6 by default) —
that's the point: it's the same result, produced two structurally
different ways. As a built-in consistency check: since the synthetic
source only ever emits one emitter's worth of pulses, the deinterleaver
should always converge to exactly one track whose `estimated_pri_us` and
`pulses` match `stats`' `mean_pri_us` and `pulses` exactly.

## The three new apps

- **Spectrogram** (`SpectrogramAnalyzer`) computes a small (8-bin, by
  default) magnitude spectrum per batch via direct correlation against a
  complex tone per bin — a Goertzel-style single-frequency DFT term rather
  than a full FFT, which is simpler to read and plenty fast at this scale.
  It rotates a unit phasor by a fixed complex multiply per sample rather
  than calling `cos()`/`sin()` for every sample (see "Optimizations"
  below for why that distinction turned out to matter a lot).

- **Jammer detector** (`JammerDetector`) flags a batch as jammed when the
  fraction of samples above a power threshold (its duty cycle) exceeds a
  configured limit — a crude stand-in for detecting sustained
  wideband/barrage jamming as opposed to normal narrow pulses. The default
  thresholds (`power_threshold=20.0,duty_cycle_threshold=0.5`) are set
  well above this repo's synthetic pulse train's actual duty cycle
  (~23%, since 120 of every 520 samples are "in pulse") specifically so
  the demo shows `0/N batches flagged` — proof the detector isn't just
  crying wolf on ordinary pulsed traffic, since a real jamming signature
  should look like continuous elevated energy, not a normal periodic pulse
  train.

- **Deinterleaver** (`Deinterleaver`) is a simplified sequential-PRI
  deinterleaver: it matches each incoming pulse against every existing
  track's predicted next-pulse time (that track's last pulse plus its
  running PRI estimate) and assigns it to the closest match within
  tolerance, or starts a new track if nothing fits. Against this repo's
  single-emitter synthetic source it always converges to exactly one
  track — see the consistency check in "Running" above.

All three follow the same shape as `PulseStatsAccumulator`: a
`Process(batch, *out)` call that folds one batch into running state and
writes the cumulative summary so far into `*out`, which is why they slot
into the existing plugin/service pattern with no changes to `module_api.h`
beyond adding their typed function-pointer signatures.

## A gotcha this example ran into (and why the build looks the way it does)

protobuf-generated message code registers its message descriptors in a
process-global registry the first time it's loaded. `monolith_app` links
the shared business logic directly *and* `dlopen()`s five plugin modules
that link the same logic — if the generated `pulse.pb.cc` code were
statically linked into each of those six places, all six would try to
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

### Avoiding a trig-call-per-sample spectrogram

The first working version of `SpectrogramAnalyzer::Process()` computed
each bin's correlation the direct way: for every sample, `phase =
omega * sample_index`, then call `std::cos(phase)` and `std::sin(phase)`.
That's correct, but at 8 bins × 4096 samples/batch × 2 calls, a single
1000-pulse run makes about 8.4 million transcendental function calls —
enough to take the monolith's benchmark from ~12ms to ~92ms almost
entirely inside one module, swamping any signal about the actual
architectural comparison this repo exists to make.

The fix: since consecutive samples within a batch have consecutive
`sample_index` values, the per-sample phasor `e^{-j*omega*n}` can be
computed by rotating a running complex value by one fixed `e^{-j*omega}`
multiply per sample instead of recomputing `cos`/`sin` from scratch each
time. That's ~4 transcendental calls per (bin, batch) pair instead of
~8000, and dropped the monolith's 1000-pulse benchmark back to ~27ms.
Output is unaffected — floating-point drift in the rotator's magnitude
over one 4096-sample batch is far below anything visible in the printed
summaries (each batch reseeds the phasor from scratch, so drift never
accumulates across batches).

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
build, from *before* any of the four consumer services are even launched
to after all five processes have exited, so process-startup cost is
counted symmetrically on both sides.

Representative result with all five apps (this machine, two separate
50-run measurements landed in the same range):

```
Results over 50 runs, 1000 pulses/run:

            modular monolith  min=26.3ms  median=27.1ms  mean=27.5ms  stdev=1.2ms  max=33.5ms
               microservices  min=51.0ms  median=55.9ms  mean=57.5ms  stdev=5.8ms  max=83.5ms

microservices mean is ~2.1x the modular monolith mean
```

The monolith, doing all five modules' work through plain function calls
with no serialization, is roughly twice as fast as the microservice build.
That gap is *wider* than it was with just detector+stats (~1.7x, see
below) for a specific reason: `spectrogram_service` and `jammer_service`
both need the raw `IQBatch`, which is the largest message in the pipeline
(4096 samples/batch) — so `detector_service` now has to serialize and
send it twice per batch (once per consumer) in addition to the
`PulseEventBatch` it was already sending to two consumers. The monolith
pays none of that; its modules just read the same in-memory `iq_batch`
directly. Adding modules that need the raw data costs the microservice
build measurably more than it costs the monolith, which is itself a real
architectural lesson: fan-out is close to free in-process and not free at
all across a wire.

**Before the three new apps**, with just detector+stats, the same
benchmark measured:

```
            modular monolith  min=10.8ms  median=11.7ms  mean=12.1ms  stdev=1.4ms  max=17.0ms
               microservices  min=18.5ms  median=20.1ms  mean=20.1ms  stdev=0.8ms  max=22.8ms
```

**And before *that***, an earlier version of `module_api.h` made the
monolith's modules speak bytes instead of typed references, on the theory
that it would make the two architectures more directly comparable — same
bytes crossing both kinds of boundary. At the same 1000-pulse/2-module
workload, that version's monolith actually *lost* to the microservices
(mean ~26.7ms vs ~19.6ms), because forcing the monolith to
serialize/deserialize every `IQBatch` just to hand it across an in-process
`dlopen()` boundary cost more than the microservices' extra process and
socket overhead did at that batch size. Passing typed references instead
(see "Eliminating the monolith's serialization tax" above) removed that
unnecessary cost and restored the expected ordering.

Small workloads still favor the monolith even more heavily than any of
this — run `./scripts/benchmark.sh 20 0` (0 pulses, i.e. one nearly-empty
batch per app) and the monolith wins by roughly 2x, since starting one
process beats starting five processes plus four TCP handshakes regardless
of how the modules talk once they're up.

### Correctness

Every change in this repo's history was re-verified the same way: run
both binaries at 6 pulses (the original default) and 1000 pulses and diff
the printed output. The monolith and microservice builds have produced
byte-identical summary statistics, spectrogram bins, jam-detection state,
and deinterleaved tracks in every case, including after adding the three
new apps and after the spectrogram phasor-rotation fix.
