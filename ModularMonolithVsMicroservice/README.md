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
IQ samples -> detector -> spectrogram -> jammer -> stats -> deinterleaver
```

- **detector** — amplitude-threshold pulse detection (reads the batch's raw
  samples, writes the pulses it found)
- **spectrogram** — a small-bin magnitude spectrum estimate per batch
- **jammer** — flags batches with sustained high-power duty cycle
- **stats** — running summary statistics over detected pulses (count, mean
  peak amplitude, mean duration, mean pulse-repetition interval)
- **deinterleaver** — groups pulses into candidate emitter tracks by PRI

The order isn't arbitrary: `detector`, `spectrogram`, and `jammer` are the
three stages that read the batch's raw samples, so they're grouped first;
`stats` and `deinterleaver` only need the pulses `detector` already found,
so they run after. That grouping is what lets the microservice build drop
the raw samples from the frame once `jammer` (the last of the three) is
done with them, instead of carrying the largest field in the pipeline
through hops that never touch it — see "Optimizations" below. Small enough
to read start to finish, but with enough sequential stages to show what
"insert a module into the pipeline" costs in each architecture.

**Scale:** the synthetic signal is 1,000,000 pulses per second (a pulse
every microsecond) split into 1000-microsecond (1ms) buffers, so every
buffer carries exactly 1000 pulses. That comes from a 10,000,000 Hz
(10 MSps) sample rate with a 10-sample pulse period (2 samples pulse, 8
samples gap — see `common/include/iq_source.h`, which documents why those
specific constants are the ones that make the arithmetic come out exact).
`./build/bin/monolith_app ./build/bin` with no other arguments generates
exactly one buffer (1000 pulses) by default; passing `1000000` generates a
full second's worth (1000 buffers). `PulseSummary.mean_pri_seconds` is a
good sanity check that the scale is configured correctly: it should always
come out to `0.000001` (1 microsecond), matching 1,000,000 pulses/sec
exactly.

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
message type, defined once, at every link in the chain either way. What
differs is what it has to go through to cross each architecture's stage
boundary. `monolith_app`'s modules run in the same process and, as of this
build, share one loaded copy of the generated protobuf code (see the
shared `pulse_proto` library below), so each stage just mutates the frame
in place through a plain function call — no serialize, no parse, and no
reason to care what else is populated on the frame at the time. The five
microservices are separate processes with separate address spaces; there
is no way to hand one a pointer into another's memory, so every link in
the chain *must* serialize the frame onto a socket. Unlike the monolith,
that means each microservice *does* care what's still on the frame when
it forwards — carrying a field no downstream stage will ever read is pure
waste on a real wire, even though it's free in-process (see
"Optimizations" below for how this build avoids that waste). That
asymmetry is not an implementation gap in this demo, it's the actual,
unavoidable cost of a real process boundary — which is exactly the
trade-off this example exists to make visible.

`detector_service` is the chain's producer: it generates the synthetic IQ
stream and runs detection locally (same as the monolith's detector
module), then sends the frame one hop downstream to `spectrogram_service`.
Each service after that — `spectrogram_service`, `jammer_service`,
`stats_service` — is *both* a TCP server (accepting the connection from
the stage before it) and a TCP client (connecting to the stage after it):
it reads a frame, enriches its own field, and forwards the frame on (after
dropping whatever fields it knows nothing downstream needs — see
"Optimizations"). `deinterleave_service`, last in the chain, only accepts;
it has nothing downstream to forward to. `monolith_app` runs the
equivalent chain as five sequential function calls in a loop over the
same in-memory frame.

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
    pulse_spectrogram_plugin.cpp   -> libpulse_spectrogram_plugin.so (chain stage 2)
    pulse_jammer_plugin.cpp        -> libpulse_jammer_plugin.so     (chain stage 3)
    pulse_stats_plugin.cpp         -> libpulse_stats_plugin.so      (chain stage 4)
    pulse_deinterleaver_plugin.cpp -> libpulse_deinterleaver_plugin.so (chain stage 5)
  host/monolith_main.cpp     Single executable; dlopen()s all five modules, runs them as a chain

microservice/               Five-executable build, wired as a linear chain over TCP
  net/framing.{h,cpp}           Minimal length-prefixed protobuf-over-TCP framing
  detector_service/main.cpp     Chain stage 1 (producer only): detects pulses, connects downstream to spectrogram_service
  spectrogram_service/main.cpp  Chain stage 2: accepts detector_service, forwards to jammer_service
  jammer_service/main.cpp       Chain stage 3: accepts spectrogram_service, forwards to stats_service,
                                  clears the raw IQ batch from the frame first (last stage that needs it)
  stats_service/main.cpp        Chain stage 4: accepts jammer_service, forwards to deinterleave_service
  deinterleave_service/main.cpp Chain stage 5 (sink only): accepts stats_service, doesn't forward

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
./build/bin/monolith_app ./build/bin 1000        # one buffer (the default)
./build/bin/monolith_app ./build/bin 1000000     # one full second at 1,000,000 pulses/sec
```

**Microservices** (five processes chained over TCP: `detector_service ->
spectrogram_service -> jammer_service -> stats_service ->
deinterleave_service`). Each listens at `base_port + offset`
(`spectrogram_service` at `+0`, `jammer_service` at `+1`, `stats_service`
at `+2`, `deinterleave_service` at `+3`) and, except for
`deinterleave_service`, connects downstream to `base_port + offset + 1`
at startup. Because each service (other than the producer) must connect
downstream *before* it can accept upstream, they have to start in the
**reverse** of data-flow order — `scripts/run_microservices.sh` handles
this for you; doing it by hand means starting the five terminals below
bottom-to-top.

```sh
./scripts/run_microservices.sh
# or directly, in five terminals, started in this order (deinterleave first):
./build/bin/deinterleave_service <base_port + 3>
./build/bin/stats_service <base_port + 2> <host> <base_port + 3>
./build/bin/jammer_service <base_port + 1> <host> <base_port + 2>
./build/bin/spectrogram_service <base_port> <host> <base_port + 1>
./build/bin/detector_service <host> <base_port> [num_pulses]

./build/bin/deinterleave_service 20054
./build/bin/stats_service 20053 127.0.0.1 20054
./build/bin/jammer_service 20052 127.0.0.1 20053
./build/bin/spectrogram_service 20051 127.0.0.1 20052
./build/bin/detector_service 127.0.0.1 20051 1000        # one buffer (the default)
./build/bin/detector_service 127.0.0.1 20051 1000000     # one full second at 1,000,000 pulses/sec
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
IQ data (`num_pulses` rectangular pulses buried in noise, 1000 by default
— one full buffer at this repo's scale) — that's the point: it's the same
result, produced two structurally different ways. As a built-in
consistency check: since the synthetic source only ever emits one
emitter's worth of pulses, the deinterleaver should always converge to
exactly one track whose `estimated_pri_us` and `pulses` match `stats`'
`mean_pri_us` and `pulses` exactly.

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
  (20%, since 2 of every 10 samples per pulse period are "in pulse" --
  see "Scale" near the top of this document) specifically so the demo
  shows `0/N batches flagged` — proof the detector isn't just
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

### Avoiding unnecessary data on the wire

Once the pipeline became a chain (see "The point of the example" above),
every hop forwarded the *entire* `pulse::PipelineFrame` regardless of what
the next stage actually read from it: `stats_service` and
`deinterleave_service` never touch `frame.iq()`, yet it rode along to both
of them anyway, and each stage's own summary field kept being serialized
and sent to stages that had no use for it.

Two changes fixed this:

1. **Reordered the chain so the three raw-sample consumers run first.**
   `detector`, `spectrogram`, and `jammer` all read `frame.iq()` — the
   largest field in the pipeline by far — so they're grouped at the front
   of the chain (`detector -> spectrogram -> jammer -> ...`) instead of
   being split across it. That means there's exactly one point in the
   chain where the raw samples are done being useful: right after
   `jammer`, the last of the three.
2. **Clear each field once nothing downstream needs it.**
   `jammer_service` calls `frame.clear_iq()` before forwarding to
   `stats_service`, since neither `stats_service` nor
   `deinterleave_service` ever reads it. Every forwarding stage
   (`spectrogram_service`, `jammer_service`, `stats_service`) also clears
   its *own* summary field right after using it locally, since no stage
   ever reads another stage's summary — only the reordering was needed
   for `iq`, since it's read by three stages spread across the chain, but
   each summary is written and consumed by exactly one.

The effect is concrete and measurable. At this repo's current scale (see
"Scale" near the top of this document — 10,000 samples/batch, 1000 pulses
per batch), a `pulse::IQBatch` for one batch serializes to about 264,828
bytes when bundled with its batch's `PulseEventBatch` (~34,941 bytes on
its own — at 1000 pulses/batch nearly every pulse produces a detected
event, so `events` is a real fraction of the frame here, not the ~0.3%
of it it was at the smaller pulse rate this fix was first measured
against). Before this fix, all four hops carried something close to the
full ~265KB frame. After it, only the first two hops
(`detector->spectrogram`, `spectrogram->jammer`) do — the last two hops
(`jammer->stats`, `stats->deinterleave`) carry only `events`, at roughly
35KB (about 13% of the unstripped size). Over a full 1,000,000-pulse
(1-second) run, that's about 460MB of raw samples no longer serialized,
transmitted, and parsed on hops that never needed them.

That's a large reduction in bytes moved, but a much smaller reduction in
wall-clock time (see "Benchmark" below) — on a loopback socket, bandwidth
is nowhere near the bottleneck; the fixed per-message cost (syscalls,
buffer allocation, protobuf parse/serialize overhead per field) dominates
far more than the marginal cost of extra kilobytes per message does. This
optimization would matter far more on a real network between real hosts,
where bandwidth and per-byte latency are actually scarce — which is
itself worth noting as a general lesson: "avoid sending unnecessary data"
pays off in proportion to how expensive bytes actually are on the wire
you're using, and loopback is about as cheap as wires get.

### Everything else

Both architectures also share a few more ordinary tuning passes:

- **Bigger batches.** `SyntheticIQSource`'s batch size went from 256 to
  4096 samples (`common/include/iq_source.h`), for the reason below, then
  later to today's 10,000 samples/batch (1000 microseconds/buffer) when
  the signal rate was scaled up to 1,000,000 pulses/sec (see "Scale" near
  the top of this document) -- that later change was driven by the target
  buffer duration, not by this optimization, but it keeps the same
  benefit. Every batch costs one fixed overhead — a protobuf
  serialize/parse, one module call or one socket message — so processing
  the same amount of data in fewer, larger batches amortizes that fixed
  cost over more work. This helps the monolith some and the microservices
  a lot, since a socket round trip's fixed cost (syscalls, kernel copies)
  is much larger than an in-process call's.
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

## Benchmark: 50 runs, 1,000,000 pulses (1 second at the current scale)

`scripts/benchmark.sh` builds both architectures, then times 50
independent, fresh-process runs of each against a synthetic input
(`./scripts/benchmark.sh 50 1000000`, which is also the default -- one
full second at this repo's 1,000,000-pulse/sec, 1000-pulses/buffer scale,
i.e. 1000 buffers per run). Each run is timed end to end: process start
to process exit for `monolith_app`; for the microservice chain, from
*before* `deinterleave_service` (the first process started, since startup
runs in reverse of data-flow order) is even launched to after all five
processes have exited, so process-startup and connection-setup cost is
counted symmetrically on both sides.

Representative result (this machine, two separate 50-run measurements
landed in the same range):

```
Results over 50 runs, 1000000 pulses/run:

            modular monolith  min=438.7ms  median=445.7ms  mean=447.6ms  stdev=7-9ms  max=477-492ms
               microservices  min=1223ms   median=1263ms   mean=1269-1276ms  stdev=35-44ms  max=1352-1499ms

microservices mean is ~2.83-2.85x the modular monolith mean
```

That ratio is narrower than the ~3.2x measured at the smaller, 1000-pulse
(one-buffer) scale below -- because the fixed costs that dominate a
single small run (process startup, sequential connection setup) stay
roughly constant while the actual work (1000 buffers instead of 1-2)
scales up, so at a full second's worth of data the fixed costs matter
proportionally less and the comparison converges toward whatever the
steady-state per-batch cost ratio between the two architectures actually
is. Neither number is "more correct" -- they're testing different things:
the small-scale numbers isolate startup/connection overhead, the
full-second numbers show what the architectural choice costs once that
overhead is amortized.

### At the smaller, one-buffer (1000-pulse) scale

This repeats the same benchmark at `./scripts/benchmark.sh 50 1000`
(1000 pulses = one buffer, no batching-across-buffers involved) for
comparison against the history below, which was all measured at small
scale before this repo's signal rate was scaled up to 1,000,000
pulses/sec (see "Scale" earlier in this document) -- so these are the
current code's numbers at the *same* scale the older entries used, not a
mix of old code and new scale:

```
            modular monolith  min=26.6ms  median=27.0ms  mean=27.3ms  stdev=0.7ms  max=31.3ms
               microservices  min=81.5ms  median=85.9ms  mean=87.5ms  stdev=6.8ms  max=122.1ms

microservices mean is ~3.2x the modular monolith mean
```

That's an improvement over the unoptimized chain (~93.7ms mean, ~3.4x —
see below), but a modest one relative to the ~24MB/run of raw samples the
previous section shows this avoids moving. That's expected, not a sign
the fix didn't work (correctness was reverified identically — see below):
on a loopback socket, moving an extra few hundred KB across two hops costs
low-single-digit milliseconds at most, so it was never going to be the
dominant cost. What still dominates the microservice build's time at this
scale, unchanged by that fix:

1. **Sequential connection setup instead of parallel.** Each service's
   downstream connection depends on the *previous* service already being
   up (see `scripts/run_microservices.sh`'s reverse-order startup), so
   five process starts and four TCP handshakes happen one after another
   instead of concurrently. This was also true of the fan-out topology's
   *accept* side, but fan-out let `detector_service` open all four of
   *its* connections in parallel rather than daisy-chaining through four
   intermediaries first.
2. **Per-message fixed costs**, still paid twice as often on the two hops
   that still carry the frame with `events` on it (`jammer->stats`,
   `stats->deinterleave`) as they were before: a `recv()`/`send()` pair of
   syscalls, a length-prefix parse, and a protobuf `ParseFromString`/
   `SerializeToString` call each carry a fixed cost that doesn't scale
   down just because the payload got smaller.

The monolith is immune to both: its "connections" are function calls that
either all succeed instantly or don't, and passing `frame` by reference
between chain stages costs nothing extra no matter how many fields ride
along unused — which is exactly why "avoid sending unnecessary data" is a
microservices-specific optimization in this repo, not a general one.

The four entries below predate this repo's scale-up to 1,000,000
pulses/sec (see "Scale" near the top of this document) and were all
measured against the *old* generator constants -- a 1,000,000 Hz sample
rate, 520-sample pulse periods, 4096-sample batches -- which no longer
exist in the code. They're kept as a historical record of each fix's own
effect at whatever scale existed when it was made, not as a byte-for-byte
comparable baseline against the "one-buffer" entry above them (which runs
*current* code, just at a small `num_pulses`).

**Before this optimization**, with the same chain topology but every hop
forwarding the full frame regardless of what the next stage read, the
benchmark measured:

```
            modular monolith  min=26.3ms  median=27.0ms  mean=27.2ms  stdev=0.8ms  max=31.5ms
               microservices  min=86.3ms  median=92.3ms  mean=93.7ms  stdev=6.4ms  max=123.3ms
```

**Before the chain topology**, with `detector_service` as a hub sending
directly to four independent consumers, the same 5-app benchmark measured:

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
buffer) against *current* code and the monolith wins by roughly 5x
(~4.3ms vs ~22.8ms), since starting one process beats starting five
processes plus four sequential TCP handshakes regardless of how much data
ends up moving once they're up. That gap is wider than the fan-out-era
~2x this same test used to show, because a chain's connection setup is
inherently sequential (see point 1 above) while fan-out's was parallel --
so a chain pays more fixed startup cost precisely when there's the least
actual work to amortize it against.

### Correctness

Every change in this repo's history was re-verified the same way: run
both binaries at a small pulse count and a larger one and diff the
printed output. The monolith and microservice builds have produced
byte-identical summary statistics, spectrogram bins, jam-detection state,
and deinterleaved tracks in every case, including after adding the three
new apps, after the spectrogram phasor-rotation fix, after converting the
fan-out topology into a linear chain, after reordering that chain and
clearing unused fields between hops, and after scaling the signal rate up
to 1,000,000 pulses/sec. Clearing a field a stage already consumed and
copying its value into a local variable first (see
`spectrogram_service`/`jammer_service`/`stats_service`'s `last_summary`
pattern) is exactly the kind of change that's easy to get backwards --
clear-then-read instead of read-then-clear silently zeroes out the value
you meant to print -- so this one got the same before/after diff
treatment as everything else. The scale-up specifically was checked
against `PulseSummary.mean_pri_seconds` coming out to exactly `0.000001`
(confirming the 1,000,000-pulse/sec rate) and the deinterleaver's single
track's `pulse_count`/`estimated_pri_seconds` matching `stats`' exactly,
at both 1000 pulses (one buffer) and 1,000,000 pulses (one full second).
