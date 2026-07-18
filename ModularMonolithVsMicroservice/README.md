# Modular Monolith vs. Microservices — a minimum viable comparison

A minimal, working C++ example that implements **the same signal-processing
pipeline twice**, once as a modular monolith and once as a set of
microservices, so the architectural trade-off is visible in code rather than
in the abstract.

The pipeline is a toy signal-intelligence front end with five stages, wired
as a **linear chain** rather than fanning out from a hub — each stage reads
and enriches one shared frame, then passes it to the next, the same way
Unix pipes chain filters:

```
IQ samples -> detector -> stats -> deinterleaver -> spectrogram -> jammer
```

- **detector** — amplitude-threshold pulse detection (reads the batch's raw
  samples, writes the pulses it found)
- **stats** — running summary statistics over detected pulses (count, mean
  peak amplitude, mean duration, mean pulse-repetition interval)
- **deinterleaver** — groups pulses into candidate emitter tracks by PRI
- **spectrogram** — a small-bin magnitude spectrum estimate per batch
- **jammer** — flags batches with sustained high-power duty cycle

Every stage after the first receives the frame the stage before it already
touched; `stats` and `deinterleaver` read the pulses `detector` found,
while `spectrogram` and `jammer` read the raw samples that have been
riding along in the frame since the top of the chain, untouched by the
stages in between. Small enough to read start to finish, but with enough
sequential stages to show what "insert a module into the pipeline" costs
in each architecture.

## The point of the example

Both variants are built from the **exact same business logic** and the
**exact same protobuf message schema** (`proto/pulse.proto`), wired in the
**exact same chain order**. Only the boundary between adjacent stages
changes:

| | Modular monolith (`monolith_app`) | Microservices (5-stage chain) |
|---|---|---|
| Process model | 1 process | 5 processes |
| Module loading | `dlopen()` at startup | separate binaries, started independently, reverse-chain order |
| Stage boundary | typed C++ reference (`module_api.h`) | TCP socket, one connection per link in the chain |
| Data crossing each boundary | a `pulse::PipelineFrame&`, in place, no copy | the whole serialized `pulse.PipelineFrame`, length-prefixed on the wire |
| Failure domain | one module crashing takes down the process | one service crashing breaks the chain at that link; upstream stages don't know until their write fails |
| Deployment | ship one executable + five `.so` files, versioned together | ship/scale/deploy each service independently |

The `pulse.PipelineFrame` envelope is what proves the point: it's the same
message, carrying the same fields, at every link in the chain either way.
What differs is what it has to go through to cross each architecture's
stage boundary. `monolith_app`'s modules run in the same process and, as
of this build, share one loaded copy of the generated protobuf code (see
the shared `pulse_proto` library below), so each stage just mutates the
frame in place through a plain function call — no serialize, no parse. The
five microservices are separate processes with separate address spaces;
there is no way to hand one a pointer into another's memory, so every link
in the chain *must* serialize the frame onto a socket, even the fields a
given stage never reads (see "Benchmark" below for what that costs).
That's not an implementation gap in this demo, it's the actual,
unavoidable cost of a real process boundary — which is exactly the
trade-off this example exists to make visible.

`detector_service` is the chain's producer: it generates the synthetic IQ
stream and runs detection locally (same as the monolith's detector
module), then sends the frame one hop downstream to `stats_service`. Each
service after that — `stats_service`, `deinterleave_service`,
`spectrogram_service` — is *both* a TCP server (accepting the connection
from the stage before it) and a TCP client (connecting to the stage after
it): it reads a frame, enriches its own field, and forwards the whole
frame on. `jammer_service`, last in the chain, only accepts; it has
nothing downstream to forward to. `monolith_app` runs the equivalent
chain as five sequential function calls in a loop over the same
in-memory frame.

## Layout

```
proto/pulse.proto          Shared message schema (IQSample, PulseEvent, PulseSummary,
                            SpectrogramSummary, JamSummary, EmitterTrack/DeinterleaveSummary,
                            and PipelineFrame -- the envelope that carries all of them through the chain)

common/                    Architecture-agnostic core ("pulsecore")
  include/pulse_detector.h   Amplitude-threshold pulse detector
  include/pulse_stats.h      Running summary-statistics accumulator
  include/spectrogram.h      Small-bin magnitude spectrum estimator
  include/jammer.h           Duty-cycle-based jamming detector
  include/deinterleaver.h    PRI-based emitter track grouping
  include/iq_source.h        Deterministic synthetic IQ generator (same input, both builds)
  src/...

monolith/                  Modular monolith build
  include/module_api.h       One typed dlsym()-able ABI every plugin module exports
                              (create/process/destroy -- process takes/mutates a PipelineFrame&)
  plugins/
    pulse_detector_plugin.cpp      -> libpulse_detector_plugin.so   (chain stage 1)
    pulse_stats_plugin.cpp         -> libpulse_stats_plugin.so      (chain stage 2)
    pulse_deinterleaver_plugin.cpp -> libpulse_deinterleaver_plugin.so (chain stage 3)
    pulse_spectrogram_plugin.cpp   -> libpulse_spectrogram_plugin.so (chain stage 4)
    pulse_jammer_plugin.cpp        -> libpulse_jammer_plugin.so     (chain stage 5)
  host/monolith_main.cpp     Single executable; dlopen()s all five modules, runs them as a chain

microservice/               Five-executable build, wired as a linear chain over TCP
  net/framing.{h,cpp}           Minimal length-prefixed protobuf-over-TCP framing
  detector_service/main.cpp     Chain stage 1 (producer only): detects pulses, connects downstream to stats_service
  stats_service/main.cpp        Chain stage 2: accepts detector_service, forwards to deinterleave_service
  deinterleave_service/main.cpp Chain stage 3: accepts stats_service, forwards to spectrogram_service
  spectrogram_service/main.cpp  Chain stage 4: accepts deinterleave_service, forwards to jammer_service
  jammer_service/main.cpp       Chain stage 5 (sink only): accepts spectrogram_service, doesn't forward

scripts/run_monolith.sh       Build + run the monolith
scripts/run_microservices.sh  Build + run the five-service chain
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

**Microservices** (five processes chained over TCP: `detector_service ->
stats_service -> deinterleave_service -> spectrogram_service ->
jammer_service`). Each listens at `base_port + offset` (`stats_service`
at `+0`, `deinterleave_service` at `+1`, `spectrogram_service` at `+2`,
`jammer_service` at `+3`) and, except for `jammer_service`, connects
downstream to `base_port + offset + 1` at startup. Because each service
(other than the producer) must connect downstream *before* it can accept
upstream, they have to start in the **reverse** of data-flow order —
`scripts/run_microservices.sh` handles this for you; doing it by hand
means starting the five terminals below bottom-to-top.

```sh
./scripts/run_microservices.sh
# or directly, in five terminals, started in this order (jammer first):
./build/bin/jammer_service <base_port + 3>
./build/bin/spectrogram_service <base_port + 2> <host> <base_port + 3>
./build/bin/deinterleave_service <base_port + 1> <host> <base_port + 2>
./build/bin/stats_service <base_port> <host> <base_port + 1>
./build/bin/detector_service <host> <base_port> [num_pulses]

./build/bin/jammer_service 20054
./build/bin/spectrogram_service 20053 127.0.0.1 20054
./build/bin/deinterleave_service 20052 127.0.0.1 20053
./build/bin/stats_service 20051 127.0.0.1 20052
./build/bin/detector_service 127.0.0.1 20051 1000
```

(Ports default to the 20000s rather than the more obvious 50000s because
Linux's ephemeral port range is typically 32768-60999 — check
`/proc/sys/net/ipv4/ip_local_port_range` — and every service in this
chain also makes an outbound connection, which claims a random port from
that range. A listener bound inside it can occasionally lose a bind()
race against one of its own chain's outbound connections. Picking
explicit ports below the ephemeral range sidesteps that entirely.)

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
`Process(input, *out)` call that folds one batch into running state and
writes the cumulative summary so far into `*out`. Every stage's plugin
wraps that call behind the single unified `pulse_stage_process` signature
in `module_api.h` (see "The point of the example" above) — reading
whichever field or fields of the shared `pulse::PipelineFrame` it needs
and writing its own field back into the same frame.

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

The fix: `module_api.h`'s function-pointer types took/returned
`pulse::IQBatch&`, `pulse::PulseEventBatch&`, and `pulse::PulseSummary*`
directly instead of bytes (see the header for the full rationale; this
later evolved into today's single `pulse::PipelineFrame&` signature once
every stage shared one envelope type, but the underlying fix is the same
one: pass typed references, not serialized bytes). `monolith_main.cpp` no
longer serializes anything in its per-batch loop; the plugins no longer
parse or allocate an output buffer. This only works safely because of the
`pulse_proto` shared-library fix above — every module has the *same*
definition of every `pulse::` type, so passing them by reference across
`dlopen()` is well-defined. The microservices can't take this shortcut:
each service is a separate process, so the frame genuinely has to be
serialized to cross every link in the chain. That asymmetry — not an
implementation gap — is the actual difference the benchmark below
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
  every plugin, every microservice `main.cpp`) used to declare a fresh
  message as a local inside the loop body.
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
chain, from *before* `jammer_service` (the first process started, since
startup runs in reverse of data-flow order) is even launched to after all
five processes have exited, so process-startup and connection-setup cost
is counted symmetrically on both sides.

Representative result with the chain topology (this machine, two separate
50-run measurements landed in the same range):

```
Results over 50 runs, 1000 pulses/run:

            modular monolith  min=26.3ms  median=27.0ms  mean=27.2ms  stdev=0.8ms  max=31.5ms
               microservices  min=86.3ms  median=92.3ms  mean=93.7ms  stdev=6.4ms  max=123.3ms

microservices mean is ~3.4x the modular monolith mean
```

That's a noticeably wider gap than the fan-out topology this repo used
before (~2.1x — see below), for two compounding reasons specific to a
*chain*:

1. **Sequential connection setup instead of parallel.** In the fan-out
   design, `detector_service` opened all four downstream connections
   itself, so they could all be waited on together. In a chain, each
   service's downstream connection depends on the *previous* service
   already being up, so the five processes and four TCP handshakes have
   to come up one after another instead of concurrently — see
   `scripts/run_microservices.sh`'s reverse-order startup. That's pure
   added latency the fan-out topology didn't have.
2. **Every hop forwards the whole frame, not just what it needs.**
   `pulse::PipelineFrame` carries `iq` (the largest field, 4096
   samples/batch) all the way from `detector_service` to `jammer_service`
   — four serialize/parse round trips — even though only `detector`,
   `spectrogram`, and `jammer` actually read it; `stats_service` and
   `deinterleave_service` pay to move it along without ever touching it.
   The fan-out design sent each consumer only the message type it
   needed. A shared envelope makes the pipeline's wiring trivial (see
   "The point of the example"), but it costs strictly more bytes on the
   wire than point-to-point messages tailored per consumer would.

The monolith is immune to both: its "connections" are function calls that
either all succeed instantly or don't, and passing `frame` by reference
between chain stages costs nothing extra no matter how many fields ride
along unused.

**With the fan-out topology** (detector_service as a hub sending directly
to four independent consumers, before this became a linear chain), the
same 5-app benchmark measured:

```
            modular monolith  min=26.3ms  median=27.1ms  mean=27.5ms  stdev=1.2ms  max=33.5ms
               microservices  min=51.0ms  median=55.9ms  mean=57.5ms  stdev=5.8ms  max=83.5ms
```

**Before the three new apps**, with just detector+stats (fan-out and
chain are the same topology at two stages), the benchmark measured:

```
            modular monolith  min=10.8ms  median=11.7ms  mean=12.1ms  stdev=1.4ms  max=17.0ms
               microservices  min=18.5ms  median=20.1ms  mean=20.1ms  stdev=0.8ms  max=22.8ms
```

**And before *that***, an earlier version of `module_api.h` made the
monolith's modules speak bytes instead of typed references, on the theory
that it would make the two architectures more directly comparable. At the
same 1000-pulse/2-module workload, that version's monolith actually *lost*
to the microservices (mean ~26.7ms vs ~19.6ms), because forcing the
monolith to serialize/deserialize every `IQBatch` just to hand it across
an in-process `dlopen()` boundary cost more than the microservices' extra
process and socket overhead did at that batch size. Passing typed
references instead (see "Eliminating the monolith's serialization tax"
above) removed that unnecessary cost and restored the expected ordering.

Small workloads still favor the monolith even more heavily than any of
this — run `./scripts/benchmark.sh 20 0` (0 pulses, i.e. one nearly-empty
batch) and the monolith wins by roughly 2x, since starting one process
beats starting five processes plus four sequential TCP handshakes
regardless of how much data ends up moving once they're up.

### Correctness

Every change in this repo's history was re-verified the same way: run
both binaries at 6 pulses (the original default) and 1000 pulses and diff
the printed output. The monolith and microservice builds have produced
byte-identical summary statistics, spectrogram bins, jam-detection state,
and deinterleaved tracks in every case, including after adding the three
new apps, after the spectrogram phasor-rotation fix, and after converting
the fan-out topology into a linear chain.
