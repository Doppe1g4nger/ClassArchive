# Modular Monolith vs. Microservices — a minimum viable comparison

> ## 🏁 This is the `claude/theoretical-limits` branch
>
> Built on `claude/max-optimization` (whose banner follows below and
> whose changes are all inherited), this branch applies the last
> methods that branch priced and declined -- the ones that trade
> teaching props for speed. The five algorithms, the synthetic signal
> (bit-for-bit), and every printed *result* are still identical; what's
> gone is the wire itself:
>
> - **Shared-memory ring transport for the C++ chain**
>   (`microservice/net/framing.{h,cpp}`): each link is now a
>   single-producer/single-consumer ring in a `shm_open` segment
>   instead of a TCP socket. The same serialized `PipelineFrame` bytes
>   that used to cross the socket are memcpy'd into a mapped slot, so
>   every service's parse/serialize logic is untouched; a "send" is a
>   memcpy plus one release-store, a "receive" is the mirror image, and
>   there are **zero syscalls on the data path**. The API deliberately
>   mirrors the TCP one (Listen/Accept/Connect/Send/Recv, keyed by the
>   same port numbers, which now name `/dev/shm/pulse_ring_<port>`), so
>   the reverse-order startup invariant the steady-state measurement
>   leans on carries over unchanged, and end-of-stream is a writer-done
>   flag where TCP used FIN. Waits spin briefly then `sched_yield()` --
>   five processes share four cores here, and a pure busy-spin would
>   fight the very pipeline it's carrying.
> - **`-march=native` globally**, with `-ffp-contract=off` pinned:
>   native makes FMA available, and GCC's default contraction would
>   silently fuse `a*b+c` into one rounding and break the bit-exact
>   stages the verification story rests on. Contraction stays enabled
>   only for the one tolerance-gated file (spectrogram.cpp).
>
> Applied, measured, and **reverted** -- kept as a negative result: the
> C++ port of numpy's GF(2) jump-ahead RNG vectorization. Bit-identical
> output, but 11.8ms vs 8.8ms monolith median -- a regression. The C++
> scalar generator loop was already compiled, inline, and running at
> memory speed; the lane-staging and grid transposition the jump-ahead
> needs cost more than the sequential dependency they removed. numpy
> won with the same trick because it was escaping the *interpreter*,
> not the dependency chain. The revert is the lesson: the identical
> optimization, applied to the identical algorithm, flips sign with the
> language runtime.
>
> Correctness gates unchanged from max-opt: monolith and AVX printed
> outputs diff byte-identical against the pre-branch references; the
> chain's results are value-identical (its log lines now say
> `via shm ring` and the transport-status wording changed -- data lines
> match exactly); all 42 unit tests pass, including a rewritten framing
> round-trip test that exercises the ring's slot-capacity bound and
> drain-after-close semantics; both verify tools pass.
>
> Steady-state medians (20x50k, same machine), max-opt -> this branch:
>
> | build | max-opt | theoretical-limits | |
> |---|---:|---:|---|
> | C++ microservices | 14.9ms | **11.5ms** | 1.3x -- and min dropped 12.5 -> 7.1ms, *below* the monolith's |
> | C++ monolith | 8.8ms | 9.2ms | flat (memory-bound; `-march=native` had nothing to sell it) |
> | C++ monolith (AVX2) | 8.7ms | 9.2ms | flat |
> | Python builds | -- | unchanged | this branch touches only the C++ chain + flags |
>
> The chain's min undercutting the monolith's is the branch's headline:
> with transport near-free, the five processes genuinely overlap their
> stages (pipeline parallelism across cores), something the monolith's
> single thread structurally can't do -- the chain's remaining ~2ms
> median gap over its own min is scheduling jitter from five processes
> on four cores, not boundary cost. This is the "boundary cost
> approaches zero" endpoint the main README priced from the start: the
> architecture penalty was never the process count, it was the wire.
>
> What was traded to get here (why main/max-opt declined it): shm rings
> only work on one host, so the cross-host scaling story -- the actual
> reason to choose microservices -- is gone, along with the
> capture-the-wire teaching prop. `-march=native` binaries aren't
> portable across CPU generations. This branch is the speed ceiling of
> this architecture on this machine, not a recommendation.
>
> ---
>
> ### Round two: thread the monolith, retire bit-exactness
>
> Two constraints survived round one: the monolith stayed
> single-threaded (the main branch's charter -- threading its stages
> turns it into the very thing it's compared against) and every
> non-gated stage stayed bit-exact. Both are now gone.
>
> - **Stage-per-thread monolith**
>   (`monolith/include/threaded_pipeline.h`, wired into both monolith
>   apps): the five modules run as a pipeline over a ring of eight
>   preallocated frames -- generation+detector fused on the producer
>   thread (the chain fuses them into detector_service for the same
>   reason), spectrogram and jammer a thread each, stats+deinterleaver
>   sharing the fourth thread to match the box's four cores. Every
>   stage still sees every batch, in order, from one thread, so all
>   running state stays lock-free and per-stage arithmetic is
>   untouched; the only synchronization is one release-store/
>   acquire-load pair per stage per batch. This is the chain's pipeline
>   parallelism without the chain's serialize/parse toll -- and it's
>   worth exactly 2x: monolith median 9.2ms -> **4.3ms**.
> - **Global `-ffast-math` (C++) / `fastmath=True` (numba)**: the
>   formal end of the repo's bit-exactness guarantee. FMA contraction,
>   reassociation, and vectorized math are allowed everywhere,
>   including the synthetic source; every gate that asserted byte or
>   bit equality now asserts a tolerance instead (unit tests at 1e-12
>   relative; verify_avx_variant's detector at 1e-9 with event counts
>   and sample boundaries still exact). Measured drift: ~4e-16
>   relative, worst case, across every gate -- and every build's
>   printed output is still *character-identical* to the pre-branch
>   references, because 3-decimal output can't see femto-scale wobble.
>   The honest ledger, though: fast-math alone moved **no** headline
>   number (the C++ stages are memory- and dependency-bound, and numba
>   sat flat at 22ms) -- the win of this round is the threading;
>   fast-math just stopped charging rent for a guarantee nobody was
>   using.
> - **Two ring-transport bugs, caught by the benchmark itself** and
>   kept here as the robustness bill for leaving TCP: (1) `Connect()`
>   could mmap the segment in the window between the consumer's
>   `shm_open(O_CREAT)` and its `ftruncate`, then SIGBUS on first
>   touch of the still-zero-sized mapping -- fixed by waiting for the
>   segment to reach full size. (2) A dead peer can't break a ring the
>   way it breaks a socket -- no RST, no EOF -- so that SIGBUS crash
>   left the four downstream services yield-spinning at ~80% CPU each
>   and silently inflated every number measured after them, until a
>   too-good-to-be-true numba regression gave the game away. Every
>   blocking wait now carries a ~30s give-up deadline. TCP users never
>   see either failure because the kernel owns socket lifecycle; a shm
>   ring's lifecycle is yours, including its funerals.
>
> Final steady-state medians (20x50k, same machine, single run):
>
> | build | round 1 | round 2 | |
> |---|---:|---:|---|
> | C++ monolith | 9.2ms | **4.3ms** | 2.1x -- the threading |
> | C++ monolith (AVX2) | 9.2ms | **4.6ms** | same pipeline win |
> | C++ microservices | 11.5ms | 10.6ms | ~flat: its parallelism *is* its process boundaries |
> | Python numba | 21.9ms | 22.0ms | fastmath: nothing measurable |
> | Python numpy | 99.1ms | 107.8ms | untouched code; run-to-run drift |
> | Python mono / micro / multiproc | 713 / 380 / 377ms | 808 / 437 / 460ms | untouched code; run-to-run drift |
>
> That closes the ledger. The monolith ends 2.4x ahead of the chain:
> handed the same pipeline parallelism the chain gets from its process
> boundaries, it keeps the parallelism and drops the serialization, so
> the chain's only remaining edge -- overlap -- is gone, and its only
> remaining cost -- serialize/parse per hop -- is not. Methods still on
> no table: threading the *Python* monolith (the GIL forecloses it
> in-process; the multiproc build already is that experiment, and its
> numbers are above), float32 (its ~1e-7 precision collides with this
> pipeline's 1e-7-second PRI tolerance, risking track-structure changes
> -- a results change, not a tolerance change), and GPU offload (a
> different repo, not a rounding of this one). Everything else priced
> across five optimization rounds is either applied above or measured,
> reverted, and written down as a negative result.
>
> ---
>
> ### Round three: the signal is a given
>
> Round two's ledger reopened under one revised premise: **in the real
> system this repo caricatures, IQ samples arrive from a radio -- no
> architecture choice makes the antenna faster.** So every build now
> generates its full signal *before* the steady-state clock starts,
> and the measured region begins at detection: the first stage that is
> actually this system's job. (The price of "the input already exists"
> being literally true is that it's resident -- ~8MB at benchmark
> scale, held by whichever process feeds the pipeline.) Every
> remaining profiled cost was then attacked:
>
> - **C++ monoliths**: the pipeline harness runs stages over the
>   pre-generated frames (per-batch frames, outputs pre-sized untimed)
>   with an explicit backpressure window -- its brief absence measured
>   ~3x slower, the fastest stage sprinting ahead and every stage
>   behind it reading stone-cold memory; round two's slot ring had
>   provided the cap implicitly. With generation gone the spectrogram
>   became the pacing stage, so it's now **two half-range analyzer
>   instances on two threads** (bins are independent correlators; each
>   instance writes its own half of a pre-sized summary, the bin-0
>   owner writes the scalars -- one writer per field, no locks, and
>   the dlopen build does it by loading the same plugin twice with a
>   bin-range config). Thread pinning was tried and REVERTED: 6.2ms
>   pinned vs 4.7ms unpinned median -- on a shared container, pinning
>   traps a stage on whichever core the neighbors are loading.
> - **C++ chain**: zero-copy ring framing -- Send/Recv overloads that
>   serialize a message directly INTO the ring slot and merge-parse
>   directly OUT of it. The payload string they delete was costing
>   each hop a ~200KB zero-fill (std::string::resize before serialize)
>   plus one copy in and one copy out; profiling had memcpy+memset at
>   ~47% of the pacing service's instructions.
> - **numpy**: the two never-ported stages got ported. Stats is now
>   vectorized (with one part *better* than vectorized: the intra-batch
>   PRI sum telescopes -- consecutive-gap sums collapse to
>   (last-first)/rate, exactly, in integer sample indices -- so it's
>   O(1)); the deinterleaver went array-native (its sequential loop is
>   the algorithm -- each event's track match depends on means updated
>   by the previous event -- but the protobuf event batch built every
>   batch solely to feed both stages is gone). ~40% faster
>   like-for-like.
> - **numba**: the entire batch loop moved into one fused @njit driver
>   kernel -- profiling put the interpreted glue between per-batch
>   kernel calls at roughly two-thirds of this build's steady state.
>   And a prediction died in the A/B: round two guessed the
>   spectrogram's parallel=True handoff "costs more than it buys";
>   fused, it's 5.2ms WITH parallel vs 14ms without. The handoff was
>   never the problem -- the glue it was buried in was.
> - **Pure-Python builds**: pre-generation only. The GIL forecloses
>   in-process threading, and the multiproc build already is the
>   process-parallel experiment.
>
> Detection-through-deinterleave medians (20x50k, same machine, same
> run; round-2 numbers included generation, so columns aren't directly
> comparable -- that's the point of the recharter; round four's final
> column follows below):
>
> | build | round 2 (incl. gen) | round 3 (detection on) |
> |---|---:|---:|
> | C++ monolith | 4.3ms | **3.4ms** |
> | C++ monolith (AVX2) | 4.6ms | **3.6ms** |
> | C++ microservices | 10.6ms | 9.6ms |
> | Python numba | 22.0ms | **5.2ms** |
> | Python numpy | 107.8ms | **65.7ms** |
> | Python monolith | 808ms | 474ms |
> | Python microservices | 437ms | 333ms |
> | Python multiproc | 460ms | 349ms |
>
> The closing picture: the C++ threaded monolith processes one second
> of signal-time per ~68ms of wall time, 3.4ms per 50k pulses -- and
> **numba lands at 5.2ms, within 1.5x of C++**, the whole Python tax
> compiled away once no interpreted code remained on the per-batch
> path. The chain sits at 2.8x the monolith: with transport zero-copy
> and generation out of the picture, what remains really is the
> irreducible cost of the architecture -- serialize, hop, parse, five
> times -- against stages that now cost ~3ms total. Still on no table:
> everything round two listed, plus splitting the chain's
> detector_service (its gen+detect fusion is gone but it still paces
> the chain; another process is another hop, and the chain's problem
> is hops).
>
> ---
>
> ### Round four: three candidates from the last profile, all three kept
>
> One more profiling pass named each build's long pole; the three
> tractable ones were tried, measured, and all survived:
>
> - **numba's phase-table spectrogram**: the kernel was the build's
>   long pole (3.2ms of 4.6) and still used the phasor recursion -- a
>   loop-carried dependency no vectorizer can break -- while C++ and
>   numpy had used cached phase tables for rounds. Ported: the kernel
>   alone went **3.16 -> 1.05ms** (the predicted 3x), though the
>   driver only went 5.2 -> 4.8ms -- the freed time exposed the next
>   layer (per-batch event-array allocations and 51 parallel-region
>   entries), which is how these ledgers always end.
> - **`fixed64` sample indices on the wire**: callgrind had varint
>   coding of the two event-index arrays at ~15% of the chain's pacing
>   service's instructions; `fixed64` makes them memcpy-class like the
>   double arrays. ~9KB more wire per batch, paid to a transport that
>   is a memcpy. Modest, real: ~0.3ms off quiet-run chain medians.
> - **Phase tables in the pure-Python spectrogram** -- the round's
>   surprise. In CPython the tables' win is structural, not
>   arithmetic: factoring the per-batch phasor out of the sum turns
>   each bin's correlator into `sum(map(mul, samples, table)) * r0`,
>   which runs entirely inside the interpreter's C internals -- zero
>   interpreted bytecode per sample, where the old (already-optimized)
>   complex recursion executed two statements per sample. Python
>   monolith **474 -> 366ms**; the microservice and multiproc builds
>   share the module and dropped to 230/246ms.
>
> Final medians (20x50k, detection-through-deinterleave):
>
> | build | round 3 | round 4 |
> |---|---:|---:|
> | C++ monolith | 3.4ms | 3.3ms |
> | C++ monolith (AVX2) | 3.6ms | **3.2ms** |
> | C++ microservices | 9.6ms | 10.8ms (noisy mid-run; min 6.0) |
> | Python numba | 5.2ms | **4.8ms** |
> | Python numpy | 65.7ms | 62.1ms |
> | Python monolith | 474ms | **366ms** |
> | Python microservices | 333ms | **230ms** |
> | Python multiproc | 349ms | **246ms** |
>
> Where each build's long pole now sits, per the closing profile: the
> C++ monolith is balance-bound across four full cores (every further
> split needs a fifth); the chain's remaining cost is protobuf's own
> parse/serialize copies -- the step past that is not serializing,
> which is the monolith; numba's is its driver's allocation and
> parallel-region overhead (~2ms of glue around ~2ms of kernels);
> numpy's is interpreter dispatch spread thin across a dozen call
> sites; the pure-Python builds' is that they are pure Python, which
> is the comparison working as intended.

> ## ⚡ Inherited: the `claude/max-optimization` banner
>
> The main branch optimizes under two self-imposed constraints: the wire
> schema is frozen, and every change must keep all builds' outputs
> byte-identical (or explicitly tolerance-gated). This branch answers
> "how fast does it go when those constraints are lifted?" -- every
> method with material measured impact, applied. The architecture, the
> five algorithms, the synthetic signal (bit-for-bit), and all printed
> outputs are unchanged; what changed is representation and codegen:
>
> - **Packed columnar schema** (`proto/pulse.proto`): `IQBatch` carries
>   packed `i[]`/`q[]` arrays plus one `first_sample_index` instead of
>   10,000 heap-allocated per-sample messages; `PulseEventBatch` is five
>   parallel packed arrays. This one change removes the pointer-chasing
>   the main branch's profiling identified as the C++ monolith's
>   dominant L1d-miss source, makes parse/serialize memcpy-class, cuts
>   the wire size ~40%, and gives every language contiguous arrays.
> - **C++ spectrogram phase tables + compiler-vectorized reduction**
>   (per-file `-fassociative-math -mavx2 -mfma` -- the reduction is
>   tolerance-gated, everything else stays bit-exact and unflagged).
> - **AVX2 kernels finally on contiguous data** -- direct vector loads
>   replace the per-lane gather that capped them at scalar parity.
> - **Python bulk transfers**: one `extend()` per array per batch in and
>   out of protobuf, replacing every per-sample/per-event protobuf call.
>
> Correctness gates: all printed outputs diff identical against
> reference outputs captured from the main branch immediately before
> branching; all 42 unit tests (golden values unchanged -- same signal)
> and both verify tools pass. Same-day, same-machine steady-state
> medians (20x50k), main branch -> this branch:
>
> | build | main | max-opt | |
> |---|---:|---:|---|
> | C++ monolith | 20.2ms | **8.8ms** | 2.3x |
> | C++ microservices | 43.6ms | **14.9ms** | 2.9x (now ~1.6x its monolith) |
> | Python monolith | 1529.9ms | **711.9ms** | 2.1x |
> | Python microservices | 868.5ms | **382.0ms** | 2.3x |
> | Python multiproc | 942.6ms | **399.6ms** | 2.4x |
> | Python numpy | 197.4ms | **97.7ms** | 2.0x |
> | Python numba | 24.1ms | **24.0ms** | 1.0x (never touched protobuf) |
>
> numba's flat line is the branch's cleanest lesson: it was already
> array-native end to end, so the schema change that roughly halved
> everything else did nothing for it -- the entire branch is, in effect,
> every other build catching up to the data layout numba chose on day
> one.
>
> Methods evaluated and *not* applied, with reasons: Unix-domain or
> shared-memory transport for the chain (the remaining chain-vs-monolith
> gap is now ~5-6ms total; UDS would shave syscall overhead worth a
> couple of ms at the cost of losing the cross-host TCP story, shm rings
> would collapse transport entirely but replace the wire-format
> demonstration this repo exists to make); global `-ffast-math`
> (per-file reassociation where tolerance-gated captured the available
> win without touching bit-exact stages); threading the C++ monolith's
> stages (that's what the microservice chain *is* -- the comparison
> would collapse into itself).
>
> Everything below this banner is the main branch's README, kept intact
> as the history and rationale this branch builds on; per-file "packed
> columnar layout" comments mark each site the schema change touched.

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

python/                    Python port of the same architecture, see "Python ports" below
  pulsecore/                 Mirrors common/: same algorithms (including the phasor-rotation
                              and other optimizations below), generated pulse_pb2.py
  monolith/
    stages.py                  Mirrors monolith/plugins/*.cpp: 5 stage wrapper classes
    monolith_app.py            Mirrors monolith/host/monolith_main.cpp
  microservice/
    framing.py                 Mirrors microservice/net/framing.cpp (identical wire format)
    detector_service.py ... deinterleave_service.py   Mirror the 5 C++ services exactly
  multiproc/                 A third architecture: one process, forking five worker processes
                                wired by multiprocessing.Pipe() instead of five independent
                                executables wired by TCP -- see "A third Python architecture" below
    detector_worker.py ... deinterleave_worker.py   Mirror the 5 microservice stages' algorithms
                                and field-clearing exactly, just reading/writing pipe ends
    multiproc_monolith_app.py  Entry point: forks all five workers, wires them, prints the result
  numba_variant/              JIT-compiles pulsecore's existing scalar loops -- see "Going further" below
    kernels.py                  @njit ports of detector/spectrogram/jammer/IQ generation
    numba_monolith_app.py       Entry point: warms up the JIT, runs the timed loop, prints the result
  numpy_variant/               Rewrites detector/spectrogram/jammer as bulk numpy array ops
    kernels.py                   Vectorized ports; not bit-identical to the rest of the repo (see below)
    numpy_monolith_app.py        Entry point
    verify_numpy_variant.py      Standalone tolerance + pulse-straddling correctness check

common/include/*_avx.h, common/src/*_avx.cpp   AVX2-vectorized detector/jammer ports (C++), a
                              fourth C++ architecture alongside monolith/microservices --
                              see "Going further" below
monolith/host/avx_monolith_main.cpp   Entry point for avx_monolith_app
monolith/host/verify_avx_variant_main.cpp   Standalone tolerance + exact-match correctness check

tests/pulsecore_tests.cpp   C++ unit tests (framework-free, run via ctest) -- see "Tests" below
python/tests/               Python unit tests (stdlib unittest), mirroring the C++ suite
  test_pulsecore.py           pulsecore algorithms + framing, same golden values as the C++ suite
  test_variants.py            numba/numpy kernel parity (skips cleanly if deps missing)

scripts/run_monolith.sh              Build + run the C++ monolith
scripts/run_microservices.sh         Build + run the C++ five-service chain
scripts/run_avx_monolith.sh          Build + run the AVX2 C++ variant
scripts/verify_avx_variant.sh        Build + run the AVX2 variant's correctness check
scripts/run_tests.sh                 Build + run both unit-test suites (ctest + unittest)
scripts/benchmark.sh                 Build + time N runs of the C++ architectures, wall clock (includes startup)
scripts/gen_python_proto.sh          Regenerate python/pulsecore/pulse_pb2.py
scripts/run_python_monolith.sh       Run the Python monolith
scripts/run_python_microservices.sh  Run the Python five-service chain
scripts/run_python_multiproc.sh      Run the Python forked-worker build
scripts/run_python_numba.sh          Run the numba-JIT variant
scripts/run_python_numpy.sh          Run the numpy-vectorized variant
scripts/benchmark_python.sh          Time N runs of all four, wall clock (includes startup)
scripts/benchmark_steady_state.sh    Time N runs of all eight, steady-state only (startup excluded;
                                       AVX2/numba/numpy series skip gracefully if their
                                       prerequisites are missing)
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

The Python port additionally requires Python 3.9+ and the `protobuf`
package:

```sh
pip install -r python/requirements.txt
./scripts/gen_python_proto.sh   # generates python/pulsecore/pulse_pb2.py
```

`scripts/run_python_monolith.sh` and `scripts/run_python_microservices.sh`
both call `gen_python_proto.sh` themselves, so this step is only needed if
you're invoking the `python/` scripts directly.

`numba_variant/` and `numpy_variant/` (see "Going further" below) need
numpy and numba on top of that -- `pip install -r
python/requirements-numeric.txt` instead of `requirements.txt`. No other
build in this repo (monolith, microservice, multiproc) has this
dependency, and none of them ever will just to keep one variant's
`import` list shorter.

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

**Python monolith** and **Python microservices** work identically, just
with `.py` scripts instead of compiled binaries, no `plugin_dir`, and
their own default port block (20151+) so they don't collide with the C++
build if both happen to run at once:

```sh
./scripts/run_python_monolith.sh [num_pulses]
./scripts/run_python_monolith.sh 1000000     # one full second at 1,000,000 pulses/sec

./scripts/run_python_microservices.sh [base_port] [num_pulses]
./scripts/run_python_microservices.sh 20151 1000000
```

**Python multiproc** (one process, forking five worker processes wired by
`multiprocessing.Pipe()` — see "A third Python architecture" below):

```sh
./scripts/run_python_multiproc.sh [num_pulses]
./scripts/run_python_multiproc.sh 1000000     # one full second at 1,000,000 pulses/sec
```

All five print the same detected pulse count, summary statistics,
spectrum, jam-detection state, and emitter tracks, computed from the same
synthetic IQ data (`num_pulses` rectangular pulses buried in noise, 1000
by default — one full buffer at this repo's scale) — that's the point:
it's the same result, produced five structurally different ways. As a
built-in consistency check:
since the synthetic source only ever emits one emitter's worth of pulses,
the deinterleaver should always converge to exactly one track whose
`estimated_pri_us` and `pulses` match `stats`' `mean_pri_us` and `pulses`
exactly.

## Python ports

`python/` is a line-for-line port of the entire C++ side: the same five
algorithms, the same monolith-vs-chain architectures, the same wire
format, generated from the exact same `proto/pulse.proto`. The point
isn't to show off idiomatic Python -- it's to hold the language as close
to the only variable as this repo can manage, so a Python-vs-C++
benchmark (see below) measures language/runtime overhead on one fixed
algorithm and architecture, not "well-optimized C++ vs. a rewritten
Python design." Concretely:

- **Same algorithms, including their optimizations.** The spectrogram's
  phasor-rotation trick (see "Optimizations" below) is ported as-is
  rather than reintroduced as a naive `cos()`/`sin()`-per-sample loop.
  Reverting an optimization on one side of the comparison would measure
  "algorithmic complexity" as much as "language," which isn't the
  question being asked.
- **Same RNG, bit-for-bit.** `pulsecore/iq_source.py` reimplements the
  xorshift32 generator with explicit 32-bit masking after every
  operation (Python integers don't wrap the way C++'s `uint32_t` does),
  so both languages generate byte-identical synthetic IQ data from the
  same seed -- confirmed by diffing output, not assumed.
- **Same wire format.** `microservice/framing.py` sends the identical
  4-byte-length-prefix-plus-protobuf frames as `framing.cpp`, using the
  same `pulse.PipelineFrame` messages (compiled from `proto/pulse.proto`
  by `protoc --python_out`, the same tool the C++ build's CMake step
  calls). A Python service and a C++ service could talk to each other
  without either one knowing the other's language -- this repo doesn't
  test that combination, but it's true by construction.
- **Same "modular" story, different mechanism.** C++'s modular monolith
  loads plugins via `dlopen()`/`dlsym()` because that's what's needed to
  let independently compiled `.so` files agree on a call signature at
  runtime without sharing source. Python doesn't need any of that
  machinery to get the same property: `import` already resolves a
  separately authored module by name at runtime, which is why
  `monolith/stages.py`'s five stage classes are Python's direct analog of
  `monolith/plugins/*.cpp`'s five `.so` files, and `monolith_app.py`'s
  `import` statements are the analog of `monolith_main.cpp`'s
  `dlopen()` calls -- see `monolith_app.py`'s docstring for the fuller
  version of this point.
- **Protobuf implementation matters.** This repo's Python build uses the
  `protobuf` PyPI package's default `upb` backend (a C extension), not
  the pure-Python fallback implementation, which would make
  serialization itself (not just the per-sample loops) dramatically
  slower and muddy the comparison below. Check
  `python -c "from google.protobuf.internal import api_implementation;
  print(api_implementation.Type())"` prints `upb` if you're unsure which
  one you have.

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
   `jammer_service` empties `frame.iq` before forwarding to
   `stats_service`, since neither `stats_service` nor
   `deinterleave_service` ever reads it. Every forwarding stage
   (`spectrogram_service`, `jammer_service`, `stats_service`) also clears
   its *own* summary field right after using it locally, since no stage
   ever reads another stage's summary — only the reordering was needed
   for `iq`, since it's read by three stages spread across the chain, but
   each summary is written and consumed by exactly one. (How the
   clearing is spelled matters more than this section originally knew:
   these are now in-place `mutable_x()->Clear()` calls, not the
   generated `clear_x()` accessors, which *delete and re-allocate* the
   submessage every batch on a non-arena message — see "Profile-driven
   optimization, round 2" below for how profiling caught that.)

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

## Python vs C++

`scripts/benchmark_python.sh` times all four implementations -- C++
monolith, C++ microservices, Python monolith, Python microservices --
against the same input. Unlike `scripts/benchmark.sh`, it defaults to the
*small* scale (1000 pulses, one buffer) rather than 1,000,000: pure-Python
execution of this repo's per-sample loops (the spectrogram's
O(bins × batch_size) inner loop dominates) is roughly two orders of
magnitude slower than the equivalent C++, so a 50-run benchmark at the
full-second scale would take on the order of an hour. Full-scale numbers
below come from a small number of individually timed single runs instead.

### At small scale (50 runs, 1000 pulses)

```
Results over 50 runs, 1000 pulses/run:

            C++ monolith  min=  6.3ms  median=  6.9ms  mean=  7.0ms  stdev= 0.5ms  max=  9.0ms
       C++ microservices  min= 30.3ms  median= 33.4ms  mean= 34.9ms  stdev= 5.6ms  max= 61.8ms
         Python monolith  min=112.3ms  median=120.2ms  mean=125.0ms  stdev=14.2ms  max=200.8ms
    Python microservices  min=345.9ms  median=361.1ms  mean=369.9ms  stdev=22.4ms  max=462.2ms

Python monolith is ~17.7-17.8x the C++ monolith mean
Python microservices is ~10.6x the C++ microservices mean
C++ microservices is ~4.7-5.3x the C++ monolith mean
Python microservices is ~2.8-3.2x the Python monolith mean
(two separate 50-run measurements landed in these ranges)
```

Python is an order of magnitude slower than C++ here, which is entirely
expected: every stage's inner loop is now interpreted bytecode instead of
compiled machine code, and even with protobuf's `upb` C-extension backend
(see "Python ports" above), each field access on a message still goes
through more machinery than a raw C++ struct member read. Within Python,
the monolith beats the microservices chain by about the same kind of
margin the C++ architectures show at this scale, for the same reason: five
process starts and four sequential TCP handshakes are fixed costs that
dominate when there's almost no actual work (one buffer) to amortize them
against.

### At full scale (single runs, 1,000,000 pulses -- 1 second of signal)

```
            C++ monolith         ~430-437ms
            C++ microservices    ~1.6-1.8s
            Python monolith      ~65-74s
            Python microservices ~41-44s

Python monolith is ~160x the C++ monolith
Python microservices is ~25x the C++ microservices
C++ microservices is ~3.9x the C++ monolith
Python microservices is ~0.6x the Python monolith (i.e. FASTER)
```

**The ordering flips.** At small scale the Python monolith wins, same as
C++; at full scale the Python *microservices* build wins -- by a wide
margin, and reproduced twice. This is the opposite of every C++ result in
this document, where the monolith wins at every scale tested (see
"Benchmark" above), and it comes down to one thing C++ doesn't have:
Python's GIL (Global Interpreter Lock) confines all bytecode execution in
a single process to one CPU core at a time, no matter how many cores the
machine has (this repo's test machine has 4). The Python monolith runs
all five stages sequentially in *one* process, so it's bound to one core
for the entire run regardless of available parallelism. The Python
microservices chain is five separate OS processes, each with its own
interpreter and its own GIL -- the OS scheduler can and does run them on
different cores simultaneously, and because the chain is a genuine
pipeline (stage N can be working on batch K while stage N-1 is already
producing batch K+1), sustained throughput benefits from that real
parallelism once there's enough work to keep the pipeline full. The
process-startup and IPC costs that make microservices lose at small scale
are still there at full scale -- they just stop mattering once 1000
batches' worth of genuinely parallelizable work swamps them.

This is a real, mechanical consequence of the language, not a quirk of
this benchmark: in C++, "avoid inter-process communication" is close to a
strictly dominant strategy, since C++ threads/processes have no equivalent
of the GIL forcing single-core execution. In a GIL-bound language, that
intuition can invert once there's enough CPU-bound work, because splitting
work across OS processes escapes the GIL's one-core-at-a-time constraint
in a way splitting it across function calls within one process cannot --
"modular monolith" and "microservices" were never chosen for their
threading properties in this repo, but in Python, one of them happens to
get free multi-core parallelism as a side effect of being multiple
processes, and the other one doesn't. The magnitude and crossover point of
this effect depend on how many CPU cores are actually available -- on a
single-core machine neither architecture could benefit, since there'd be
nowhere for the extra processes to run in parallel.

### A third Python architecture: multiprocessing

The GIL explanation above raises an obvious follow-up: is *threading*
also a fix, or does it have to be separate processes? And if it's
processes either way, is the microservices build's TCP/protobuf boundary
actually buying anything, or would plain multiprocessing get the same win
for less machinery?

Threading doesn't help here -- the GIL prevents concurrent *bytecode*
execution across threads in the same process regardless of how many
threads exist, so a threaded version of `monolith_app.py` would still be
bound to one core for this CPU-bound work, the same as it is today (quite
possibly slightly worse, from lock-handoff overhead as threads trade the
GIL back and forth for no parallelism gained). Multiprocessing does help,
which is unsurprising once you notice it's the same trick the
microservices build already relies on: separate OS processes, each with
its own GIL.

`python/multiproc/` puts that trick in its most direct form: one script
(`multiproc_monolith_app.py`) that forks five worker processes --
`detector_worker.py` through `deinterleave_worker.py`, each running the
exact same algorithm and field-clearing logic as its
`microservice/*_service.py` counterpart -- and wires them together with
`multiprocessing.Pipe()` instead of TCP sockets. Structurally it's much
closer to the monolith than to the microservices build: one command, no
ports, no independently-launched executables, no reverse-order startup
dance (a `Pipe()` is created synchronously by the parent before any
worker is even forked, so there's no `bind()`/`listen()`/`accept()`/
`connect()` sequence to reason about at all). It also can't rely on a
socket's half-close to signal end-of-stream the way `framing.py` does --
every worker here is forked from the same parent *after* all four pipes
already exist, so every sibling inherits its own file descriptor for
every pipe end, not just the one or two it actually uses; real EOF would
require every one of those duplicates closed. Each worker sends an
explicit `None` down its outbound pipe instead, once it has no more
batches to forward.

What this build keeps from the microservices build is exactly the one
property the GIL explanation above says should matter: five separate
processes, five separate GILs, real multi-core parallelism once the
pipeline is full. It's designed as a controlled variant, not a competing
benchmark target -- it isolates "multiple processes" from "network
boundary" so the steady-state numbers below can show which one was
actually doing the work.

## Steady-state benchmark

Every benchmark above times each run from the *outside*: `date`/wall clock
around process start to process exit. For the microservice chain that
necessarily bundles in five process starts and four sequential TCP
handshakes (see "At the smaller, one-buffer (1000-pulse) scale" above,
point 1) alongside the actual per-batch processing work -- which is
useful for answering "how long does it take to stand this chain up and
run it once," but it conflates two very different costs: one-time setup
and repeatable steady-state throughput. `scripts/benchmark_steady_state.sh`
isolates the second one.

**Methodology.** Every one of the seventeen programs in this repo (six
C++, eleven Python -- six standalone plus five `multiproc` workers) now
self-reports its own steady-state duration on a `STEADY_STATE_MS <value>`
line, using its own local clock (`std::chrono::steady_clock` /
`time.perf_counter()`) around only the batch-processing loop:

- `monolith_app`/`monolith_app.py` start the timer right after `dlopen()`/
  `import`-ing all five modules, and stop it after the last batch has been
  processed by all five stages -- module loading and process exit are
  excluded, but all five stages' sequential work on every batch is
  included, since the monolith runs them one after another in a single
  thread.
- `detector_service`/`detector_service.py` (the chain's producer) start
  the timer right after `connect()` to `spectrogram_service` succeeds, and
  stop it after the last batch has been sent. `multiproc/detector_worker.py`
  starts its timer right at the top of its loop instead, since a
  `multiprocessing.Pipe()` needs no analogous connect step (see "A third
  Python architecture" above) -- there's no setup cost left to wait out.
- `deinterleave_service`/`deinterleave_service.py`/
  `multiproc/deinterleave_worker.py` (each build's sink) start the timer
  on their *first* successful receive and stop it on their last -- i.e.
  the span from "the first frame reaches the end of the chain" to "the
  last frame reaches the end of the chain."

That sink-side span is the number this script reports for both the TCP
microservices and the multiproc build, for a reason worth spelling out:
for the TCP build, every service (other than the producer) must connect
downstream before it can accept upstream, so the whole chain has to be
five-deep connected before `detector_service`'s own `connect()` can
succeed at all -- meaning no data can reach `deinterleave_service` any
earlier than that either. The multiproc build gets the same guarantee
more directly: `multiproc_monolith_app.py` only starts feeding it batches
after every worker has been forked and every pipe end handed off, so
nothing can reach `deinterleave_worker.py` before the whole chain exists.
Either way, the sink's first-recv-to-last-recv span is, by construction,
exactly its build's whole-pipeline steady-state drain time, with zero
cross-process timestamp correlation needed to prove it excludes setup
cost. (The middle stages and each build's producer also report their own
`STEADY_STATE_MS`, printed to stdout/discarded by this script, for anyone
who wants to see which single stage is the bottleneck -- they're just not
the number the summary table below uses.)

**Two different things are being measured.** The monolith number is the
*sum* of all five stages' work on every batch, since the monolith runs
them sequentially in one thread with no overlap between batches. The
microservices and multiproc sink numbers are *pipelined* throughput: five
OS processes run concurrently, so once the pipeline is full, stage N can
be working on batch K while stage N-1 is already producing batch K+1, and
the sink's drain rate is bounded by whichever single stage is slowest,
not by the sum of all five. That's not a flaw in the measurement -- it's
a real architectural difference between the two kinds of build, the same
one a production pipeline would actually experience -- but it does mean
this comparison isn't strictly "identical total work, architecture X pays
more overhead than architecture Y." It's closer to "sequential total cost
vs. pipelined bottleneck-bound throughput," and the pipelined numbers can
look better than naive intuition suggests for exactly that reason (see
the Python results below, where they do).

Run it with `./scripts/benchmark_steady_state.sh [runs] [num_pulses]`
(defaults: 20 runs, 50,000 pulses/run = 50 batches/run). The pulse count
matters here in a way it didn't for the wall-clock benchmarks: the sink's
first-to-last-recv span only means something statistically if it covers
many batches. Calibrating against the previous default of 1000 pulses (1-2
batches) produced wildly noisy, physically implausible per-run swings --
there just aren't enough inter-arrival gaps in the window to average over.
50 batches/run was enough to bring run-to-run variance down to a
reasonable range without making the Python runs (the slower side of this
comparison) take more than about two minutes total.

```
Steady-state results over 20 runs, 50000 pulses/run:

            C++ monolith  min=  17.197ms  median=  17.915ms  mean=  17.947ms  stdev=  0.324ms  max=  18.489ms
       C++ microservices  min=  51.217ms  median=  53.807ms  mean=  54.568ms  stdev=  3.683ms  max=  69.521ms
         Python monolith  min=1320.622ms  median=1378.771ms  mean=1436.201ms  stdev=198.614ms  max=2273.861ms
    Python microservices  min= 686.449ms  median= 711.180ms  mean= 730.053ms  stdev= 48.229ms  max= 862.144ms
        Python multiproc  min= 727.831ms  median= 764.661ms  mean= 787.934ms  stdev= 60.189ms  max= 932.656ms

Python monolith is 80.0x the C++ monolith mean
Python microservices is 13.4x the C++ microservices mean
C++ microservices is 3.04x the C++ monolith mean
Python microservices is 0.51x the Python monolith mean
Python multiproc is 0.55x the Python monolith mean
Python multiproc is 1.08x the Python microservices mean
```

(These are post-"Correctness audit and hyperoptimization pass" numbers --
see that section below for what changed and the full before/after
comparison. The qualitative shape is unchanged from before that pass, and
sharper in places, so the discussion below still holds.)

With startup and connection setup excluded, the qualitative picture from
the wall-clock benchmarks above holds up, and actually sharpens:

- **C++ microservices still cost ~3.0x the monolith**, essentially
  identical to the ~3.2x seen at the small wall-clock scale (see "At the
  smaller, one-buffer (1000-pulse) scale" above) -- which says the ratio
  measured there was never mostly a startup artifact. The per-batch
  serialize/parse/syscall cost paid at each of the chain's four hops
  really is the dominant term, not process-launch overhead riding along
  with it.
- **Both Python multi-process builds still beat the Python monolith**
  (0.51x for microservices, 0.55x for multiproc -- both roughly
  45-49% faster), the same GIL-driven crossover documented in "Python vs
  C++" above -- and here it's isolated from any contribution by
  process-startup parallelism, since none of these timers start until
  every process is already up and connected/wired. The processes
  genuinely pipeline CPU-bound work across cores once the pipeline is
  flowing; that's a steady-state effect, not a startup one.
- **Multiproc and microservices are close but no longer within noise of
  each other** (1.08x -- multiproc now consistently a bit slower). This
  is discussed in the hyperoptimization section below: with the
  pulsecore-level bottleneck cut down, the TCP build's larger message
  (protobuf bytes over a socket) and the multiproc build's smaller one
  (a pickled Python object over a pipe) now sit close enough together
  that a different, previously-masked cost -- pickling a `bytes` object
  through `Connection.send()` on every hop, see "A third Python
  architecture" above -- has become visible instead of being swamped by
  the stages' own per-sample work. It's still true that escaping the GIL
  is what buys the win over the monolith, and the network boundary isn't
  free either; there's just less headroom left between the two now for
  each one's remaining fixed costs to hide in. Where they still differ
  structurally is everything this benchmark deliberately excludes: the
  microservices build's five independent executables, ports, and
  reverse-order startup versus multiproc's one command and no network
  stack at all -- a real advantage for multiproc that a steady-state-only
  number can't show, which is exactly why "Steady-state benchmark" and
  the wall-clock benchmarks above are answering different questions on
  purpose.

## Correctness audit and hyperoptimization pass

A dedicated pass through every implementation: read every core algorithm
in both languages side by side against its counterpart, read every
wiring file (module loading, the plugin shims, both mains, both framing
layers, all five microservice mains and their Python ports, and the
multiproc workers), then optimized what was safe to optimize -- without
changing what any of the five builds compute.

**Correctness findings.** No functional or behavioral bugs turned up
anywhere -- all five builds already agreed, and still agree, on every
numeric output. What the audit did find was four stale comments, left
over from before the signal rate was scaled up to 1,000,000 pulses/sec
(see "Scale" near the top of this document), that still cited the old
4096-sample batch size or the old ~23% duty cycle after the constants
behind those numbers had changed to 10,000 samples/batch and 20%: one in
`jammer.h`'s docstring, one in `spectrogram.cpp`'s phasor-rotation
comment, one in `module_api.h`, and one in `jammer_service/main.cpp`. All
four described the code's current behavior incorrectly (not a historical
event, which this README documents deliberately elsewhere -- see "Before
this optimization" and similar callouts above), so they were factual bugs
in the comments, not just staleness, and are now fixed to match the
constants actually in force.

**C++ optimizations.**

- **Squared-magnitude threshold comparison in `PulseDetector::Process()`.**
  The original computed `sqrt(i*i + q*q)` for every sample just to
  compare it against an amplitude threshold. Since `sqrt` is monotonic
  increasing over non-negative reals and an amplitude threshold is never
  negative, `sqrt(x) >= t` iff `x >= t*t` -- so the threshold is now
  squared once in the constructor, and `sqrt()` is only called for
  samples that actually cross it (the peak/mean-amplitude bookkeeping
  still needs the real magnitude, but only for those). At this repo's
  20% duty cycle, that cuts the `sqrt()` calls in the pipeline's hottest
  per-sample loop by roughly 80%. This is not an approximation --
  verified against the unmodified Python port and against the C++
  build's own pre-change output at 1,000 and 50,000 pulses, byte-for-byte
  identical either way.
- **Link-time optimization.** `CMakeLists.txt` now enables
  `CMAKE_INTERPROCEDURAL_OPTIMIZATION` for Release builds, guarded behind
  `check_ipo_supported()` so it's a no-op on a toolchain that can't do
  it. This lets the compiler inline pulsecore's small, frequently-called
  `Process()` methods into their callers (the plugins, `monolith_main.cpp`,
  each microservice's `main.cpp`) across translation-unit boundaries --
  free performance with no source change and no semantic risk, since LTO
  only affects codegen within each already-existing final link target
  (each `.so`, each executable), never linkage or registration behavior,
  so it can't disturb the `pulse_proto` single-registration invariant
  documented above.

**Python optimizations.** Every one of these lives in `python/pulsecore/`,
which every Python build (monolith, microservices, and multiproc) imports
-- so a fix made once propagates to all three automatically, with no
per-build duplication.

- **The same squared-magnitude fix**, ported into `pulse_detector.py`
  exactly as above.
- **Reading each sample's `i`/`q` once per batch instead of once per
  bin, in `spectrogram.py`.** The phasor-rotation loop runs every
  sample through all `num_bins` correlators, and the original code read
  `s.i`/`s.q` -- protobuf-generated property accessors, not free
  attribute reads the way a C++ struct member is -- fresh from the
  message on every `(bin, sample)` pair. At 8 bins that's 8x the
  attribute-access cost for values that never change across bins. The
  fix reads every sample's `i`/`q` into plain Python lists once, before
  the bin loop, and has the per-bin correlator iterate those instead. A
  C++ compiler hoists the equivalent redundant reads automatically (that
  asymmetry is exactly why this fix has no C++ counterpart -- there was
  nothing to hoist that the compiler wasn't already hoisting).
- **Inlining the xorshift32 RNG in `iq_source.py`.** The original called
  a bound method, `self._next_noise()`, twice per sample -- 20,000 method
  calls per 10,000-sample batch just for RNG dispatch. The generator
  loop now inlines the same xorshift steps directly, using a local
  variable for the RNG state instead of a `self` attribute, matching the
  same reasoning as the spectrogram fix: a C++ compiler inlines the
  equivalent private-method calls automatically at `-O3`, so doing it by
  hand in Python is what makes the two languages' *actual* runtime
  behavior comparable, not what makes them diverge. Also precomputed the
  one nonzero I/Q component value instead of recomputing a division for
  it every sample, and replaced a per-sample `idx % period` with an
  increment-and-wrap, since the phase cycles in lockstep with the sample
  index anyway.
- **Hoisting per-iteration running state out of `self` and into local
  variables**, in `pulse_detector.py`, `pulse_stats.py`, and
  `deinterleaver.py`'s hot loops -- read once before the loop, updated
  as locals throughout, written back to `self` once at the end. CPython
  resolves a local variable (`LOAD_FAST`) faster than an attribute
  lookup on `self` (`LOAD_ATTR`), and these loops run once per sample or
  per event, so avoiding repeated attribute access on every iteration is
  a real, if smaller, win in the two lower-volume modules (`pulse_stats.py`,
  `deinterleaver.py` operate on detected pulses, roughly 1,000/batch,
  not the full 10,000-sample batch).

None of these change what's computed, only how many times the same
computation's inputs get re-fetched or re-derived -- confirmed by diffing
every build's full output (not just summary statistics) against every
other build's, at 1,000 and 50,000 pulses, after every change and again
after a full clean rebuild and proto regeneration at the end. All five
builds remain byte-for-byte identical to each other on every field, the
same standard "Correctness" below has held to throughout this repo's
history.

**Before/after.** Same benchmark, same machine, same parameters
(`./scripts/benchmark_steady_state.sh 20 50000`) as "Steady-state
benchmark" above, run once before this pass and once after:

```
                          before (mean)   after (mean)   speedup
    C++ monolith            18.405ms       17.947ms        1.03x
    C++ microservices       55.389ms       54.568ms        1.02x
    Python monolith       2352.443ms     1436.201ms        1.64x
    Python microservices  1527.788ms      730.053ms        2.09x
    Python multiproc      1473.711ms      787.934ms        1.87x
```

The C++ side moved modestly, as expected: `sqrt()` is a single pipelined
hardware instruction on any machine this is likely to run on, so avoiding
80% of them saves real but small cycles, and LTO's cross-TU inlining was
already fighting a codebase where most of the hot code was one function
per translation unit. The Python side moved dramatically -- 1.6x to just
over 2x -- because CPython's per-operation overhead (attribute lookups,
bound-method calls, redundant protobuf accessor reads) was a much larger
fraction of the total cost to begin with, exactly the kind of overhead
these fixes target. That asymmetry is itself a small illustration of the
point "Python vs C++" makes at length above: the two languages don't pay
for the same operations at the same rate, so the same class of fix lands
very differently depending which one it's applied to.

One side effect worth calling out: before this pass, Python multiproc
was slightly *faster* than Python microservices (0.96x); after, it's
slightly *slower* (1.08x -- see "Steady-state benchmark" above). Cutting
down the shared pulsecore bottleneck didn't just make both builds
faster, it changed which cost dominates what's left. `multiproc`'s
`Connection.send()`/`recv()` pickle a Python `bytes` object on every
hop (see "A third Python architecture" above for why -- it's what buys
the `None`-sentinel shutdown instead of needing every worker to track
every pipe end it doesn't use), while the TCP build sends the same raw
protobuf bytes with no extra wrapping. That pickling overhead was always
there; it just used to be too small next to the stages' own per-sample
work to move the needle. With that work now roughly halved, it's a
visible cost instead of a hidden one -- a reminder that "optimize the
common path" can un-mask a different bottleneck rather than eliminate
the bottleneck concept entirely.

## Going further: AVX2, numba, numpy

A direct follow-up to the hyperoptimization pass above: can the same
five algorithms go faster still with AVX instructions (C++) or
numpy/numba (Python)? Three more variants, one per technique, each
answering the question honestly rather than assuming yes:

- **`avx_monolith_app`** (C++) -- `common/include/pulse_detector_avx.h` /
  `jammer_avx.h` hand-vectorize the detector's threshold decision and the
  jammer's power-sum reduction with AVX2 intrinsics. Spectrogram, stats,
  and the deinterleaver are unmodified pulsecore code. Built as its own
  library (`pulsecore_avx`) and executable, guarded behind an actual
  `check_cxx_compiler_flag` check in `CMakeLists.txt` -- the original
  `pulsecore` and every target that links it are untouched, same
  "add a variant, don't replace the baseline" approach the Python side's
  multiproc/numba/numpy builds already use.
- **`python/numba_variant/`** -- JIT-compiles the *exact same* scalar,
  per-sample loops already in `pulsecore/` (detector, spectrogram,
  jammer, and IQ generation) with `@njit`, instead of rewriting them.
- **`python/numpy_variant/`** -- rewrites detector, spectrogram, and
  jammer as bulk numpy array operations instead of per-sample loops.
  IQ generation is not rewritten (see below for why).

Run them with `./scripts/run_avx_monolith.sh [num_pulses]`,
`./scripts/run_python_numba.sh [num_pulses]`, and
`./scripts/run_python_numpy.sh [num_pulses]` (the latter two need
`pip install -r python/requirements-numeric.txt` -- numpy and numba
aren't a dependency of anything else in this repo and stay that way).

### Two of these are not bit-identical to the rest of the repo, and that's expected

Every build up to this point in the README has been byte-for-byte
identical on every field, and that bar mattered enough to build
dedicated tooling around (`scripts/verify_avx_variant.sh`,
`python/numpy_variant/verify_numpy_variant.py`) rather than relax it
quietly. **numba_variant is still bit-identical** -- it compiles the same
operations in the same order, so a sequential accumulation like `re +=
...` inside a loop is still a sequential accumulation after JIT
compilation, just compiled instead of interpreted. **numpy_variant and
the AVX2 jammer are not**, and can't be while staying vectorized:
`np.sum()` and a horizontal AVX2 reduction both combine partial results
in a different order than the scalar loop's single running total
(pairwise summation, four SIMD lanes reduced at the end), and
floating-point addition isn't associative -- reordering it changes the
last few bits of the result even though every input and every operation
performed is identical. This is verified, not just asserted: both
verification scripts measure the actual relative error against the
scalar reference (numpy_variant: ~3e-11; the AVX2 jammer: ~2e-15, both
five to nine orders of magnitude below the 1e-9 tolerance either checks
against) and include a dedicated test for a pulse straddling a batch
boundary -- a case this repo's own signal never produces at its current
batch-size/period constants, but the vectorized detector code still has
to handle correctly regardless.

The **AVX2 detector stays bit-identical**, unlike the jammer, because its
vectorized part is a pure per-element computation (`i*i+q*q`, then
compare), not a reduction -- IEEE-754 guarantees a multiply or add
produces the same result regardless of what other SIMD lanes are doing,
since there's no cross-element combination whose order could change.
(It also takes real care to stay that way: the two vector ops used
--separate multiply, then add-- are deliberately *not* fused into one
FMA instruction, via `-ffp-contract=off` on that build target. An FMA
rounds once instead of twice and would silently produce a different
last bit than the scalar version's separate multiply-then-add, for
every sample.)

**The RNG and numpy: a bottleneck that moved three times.** The
xorshift32 RNG is a genuinely sequential recurrence -- each state
depends on the previous one -- which doesn't rewrite into bulk array
ops without a materially more advanced technique. `numba_variant` has
no such problem: numba JIT-compiles the sequential loop directly, no
rewrite required. That asymmetry -- one technique shrugs off a
sequential bottleneck, the other can't without real extra work -- is
one of the more interesting differences between the two approaches.
numpy_variant's generation went through three designs as profiling kept
relocating its bottleneck: first through the protobuf generator plus
per-batch array conversion (measured as its biggest cost; removed in
round 2), then array-native with only the RNG recurrence left as a
Python loop (which then *became* the biggest cost), and finally the
advanced technique after all -- GF(2) jump-ahead, once it was the only
thing left to fix. See "Vectorizing the 'unvectorizable' RNG" under
round 2 below, and `numpy_variant/iq_source_arrays.py` for the design.

### An AVX regression, caught and fixed, not just reported

The first working version of the AVX2 detector and jammer measured
*slower* than the plain scalar C++ versions -- not a rounding error, a
consistent 10-30% regression, confirmed with 15-run comparisons before
concluding it was real. The cause: that version extracted every sample's
`i`/`q`/`sample_index` into batch-sized heap arrays first, ran the
vectorized comparison as a second pass, then a third pass for the
sequential pulse state machine. protobuf's repeated `IQSample` field is
a repeated *message* field -- `RepeatedPtrField` stores pointers to
separately heap-allocated objects, not a contiguous `double[]` -- so
reading a sample's fields means chasing a pointer no matter what happens
to the value afterward. That pointer-chase, not the handful of FLOPs in
`i*i+q*q`, is this loop's actual cost, and the three-pass version paid
real extra memory traffic (writing, then re-reading, batch-sized
intermediate arrays) to vectorize arithmetic that was never the
bottleneck.

The fix: process in chunks of 4 samples using small, register-resident
scratch arrays instead of batch-sized ones, so the pointer-chase happens
exactly once per sample -- matching the scalar version's memory
behavior -- with the compare itself still vectorized. That closed most
of the gap: at this repo's scale, the AVX2 detector and jammer now land
within noise of their scalar equivalents (sometimes a few percent
faster, sometimes a few percent slower, run to run), rather than
consistently 10-30% slower.

**The honest net finding: AVX2 doesn't meaningfully help this specific
workload**, and the reason is itself the useful part of the answer. This
pipeline's per-sample work is only a few FLOPs -- nowhere near enough
arithmetic intensity to be compute-bound. Its actual cost is protobuf's
memory access pattern (one pointer dereference per sample, in an array
of pointers rather than an array of values), and SIMD instruction width
cannot fix a memory-access-pattern bottleneck -- it only makes the
arithmetic faster, and the arithmetic was never what was slow. Getting a
real win here would need a data-layout change (e.g. `repeated double`
fields instead of `repeated IQSample`, giving a genuinely contiguous,
directly SIMD-loadable buffer) -- which would mean changing
`proto/pulse.proto`, the wire format every build in this repo shares,
for the benefit of one variant. Out of proportion for what this
exploration set out to answer, so left as the honest conclusion instead:
*this* algorithm, on *this* data layout, is memory-bound, not
compute-bound, and hand-rolled AVX2 was never going to fix that no
matter how carefully it was written.

### Results

Same steady-state methodology as "Steady-state benchmark" above (timer
excludes JIT warm-up/compilation the same way it excludes `dlopen()`/
`import` elsewhere), 50,000 pulses:

```
                            mean        vs. own baseline
    C++ monolith (scalar)   21.67ms
    C++ monolith (AVX2)     21.69ms     1.00x (no meaningful change -- see above)
    Python monolith       1597.49ms
    Python numba           131.48ms     12.15x faster
    Python numpy          1209.78ms     1.32x faster
```

**numba is the standout result**, and it's a direct consequence of what
it doesn't have to give up: it compiles the *existing* algorithm as-is,
so there's no vectorization tax to pay anywhere, including IQ
generation, where numpy_variant has to fall back to the interpreted
generator specifically because numpy can't cheaply vectorize a
sequential RNG. **numpy's more modest 1.32x** is the honest cost of that
fallback plus the `pulse_pb2.IQBatch` round-trip it still pays for
detector/spectrogram/jammer's inputs (`array_view.extract_iq` is one
Python-level attribute read per sample per array -- the same class of
cost the original interpreted loop paid, not eliminated by vectorizing
what happens *after* the extraction). **AVX2's ~1.00x** is the most
surprising result of the three only if you expect vectorization to
always help; once the actual bottleneck (pointer-chasing, not
arithmetic) is identified, it's the expected one.

### The full eight-way picture

`scripts/benchmark_steady_state.sh` now runs all eight builds in one
pass (the AVX2/numba/numpy series skip gracefully, with a labeled
"skipped" row instead of a failure, when their prerequisites are
missing). One representative full run:

```
Steady-state results over 20 runs, 50000 pulses/run:

            C++ monolith  min=   21.720ms  median=   22.297ms  mean=   22.778ms  stdev=   1.367ms  max=   27.775ms
     C++ monolith (AVX2)  min=   21.470ms  median=   21.915ms  mean=   22.076ms  stdev=   0.722ms  max=   24.701ms
       C++ microservices  min=   59.305ms  median=   63.095ms  mean=   67.609ms  stdev=   9.976ms  max=   96.403ms
         Python monolith  min= 1586.407ms  median= 1646.678ms  mean= 1657.574ms  stdev=  66.499ms  max= 1797.578ms
    Python microservices  min=  790.142ms  median=  827.823ms  mean=  851.087ms  stdev=  69.912ms  max= 1108.340ms
        Python multiproc  min=  829.218ms  median=  898.560ms  mean=  932.161ms  stdev= 123.663ms  max= 1283.851ms
            Python numba  min=  130.521ms  median=  143.063ms  mean=  158.668ms  stdev=  41.993ms  max=  310.130ms
            Python numpy  min= 1176.174ms  median= 1209.414ms  mean= 1219.363ms  stdev=  35.160ms  max= 1284.947ms

Python monolith is 72.8x the C++ monolith mean
Python microservices is 12.6x the C++ microservices mean
C++ microservices is 2.97x the C++ monolith mean
Python microservices is 0.51x the Python monolith mean
Python multiproc is 0.56x the Python monolith mean
Python multiproc is 1.10x the Python microservices mean
C++ AVX2 monolith is 0.97x the C++ (scalar) monolith mean
Python numba is 0.096x the Python monolith mean (10.4x faster)
Python numpy is 0.74x the Python monolith mean (1.36x faster)
```

(numba's mean is inflated by one 310ms outlier -- its median, 143ms, is
the more representative number, consistent with the dedicated-run table
above. numba remains the fastest Python build by a wide margin -- faster
than both multi-process architectures, since it removes interpreter
overhead outright rather than trading it for multi-core parallelism.)

This run is also the "before" baseline for the second, profile-driven
optimization pass below ("Profile-driven optimization, round 2") --
the numbers in *its* table supersede these.

## Profiling: where the time actually goes

Everything above reports *how long* each build takes; this section is
about *why*, measured with cachegrind and callgrind (valgrind 3.22) for
the C++ builds and cProfile for the Python ones -- the same evidence
base behind several claims made earlier in this document, collected in
one place. All numbers below are from 50,000-pulse runs on the same
machine as the benchmarks above.

### C++: cachegrind proves the memory-bound claim

"An AVX regression, caught and fixed" above argues this workload is
bound by protobuf's pointer-chasing, not arithmetic. Cache simulation
confirms it quantitatively:

```
                          scalar monolith      AVX2 monolith
    instructions (Ir)     167.5M               166.1M
    D1 (L1d) misses       7.087M               7.087M
    D1 miss rate          21.5%                20.9%
    LL (last-level) miss  0.1%                 0.1%
```

A 21% L1-data miss rate is enormous for a numeric kernel (well-blocked
numeric code typically sits under 5%), and the AVX2 build changes
neither the instruction count nor the miss count in any meaningful way
-- the two builds are the same program as far as the memory system is
concerned, which is why they benchmark identically. The near-zero
last-level miss rate says the working set (one 10,000-sample batch of
heap-allocated `IQSample` messages) fits in L2/L3 -- so this isn't a
DRAM problem, it's pure L1-scale pointer-chasing: `RepeatedPtrField`
scatters 10,000 little messages across the heap, and every read of a
sample's `i`/`q` starts with a pointer dereference the prefetcher can't
fully hide.

Attributing the misses per function (`cg_annotate --sort=D1mr`) makes
the picture sharper still: **75.7% of all D1 read-misses land inside
`SpectrogramAnalyzer::Process`** -- it walks all 10,000 samples once per
bin, 8 bins per batch, so it pays the pointer-chase 8x per batch where
the detector and jammer (about 9.5% of misses each) pay it once. Which
also explains, in one number, why AVX-porting the detector and jammer
was never going to move the total: the two ported stages own less than
a fifth of the misses between them, and the miss-heaviest stage (the
spectrogram) is exactly the one with no portable vectorized
`cos()`/`sin()` to build on.

Per-function instruction counts (callgrind) for the scalar monolith,
for reference: spectrogram 59.7%, IQ generation 18.7% (inlined into
`main` by LTO), detector 5.4%, jammer 3.1%, deinterleaver 1.9%, protobuf
`Clear()`/malloc machinery ~5%.

### C++: the microservice chain executes 7.7x the instructions for the same work

Running each of the five services under callgrind during one chain run
and summing:

```
    detector_service       151M Ir   (~40M generation/detection, the rest serialize)
    spectrogram_service    601M Ir   (~100M spectrogram -- same as the monolith's -- the rest parse/serialize/malloc)
    jammer_service         436M Ir   (~5M jammer work; >98% parse/serialize/malloc)
    stats_service           57M Ir
    deinterleave_service    47M Ir
    ------------------------------
    chain total          1,292M Ir   vs. the monolith's 167M for identical output
```

The excess is almost entirely boundary cost, itemized: `IQSample::
_InternalParse` / `_InternalSerialize` / `ByteSizeLong`, `WireFormatLite
::InternalWriteMessage`, and -- the single biggest line item in the two
iq-carrying middle services -- `_int_malloc`/`_int_free`/
`malloc_consolidate`, because parsing a frame re-materializes 10,000
heap-allocated `IQSample` objects *per hop* and tears them down again
after forwarding. This is the instruction-level anatomy of the ~3x
steady-state gap the benchmarks show, and of the wire-format asymmetry
"The point of the example" describes: the monolith passes one frame by
reference; every microservice hop rebuilds and destroys it.

### Python: cProfile across the builds

**Monolith** (2.85s under profiler): spectrogram `process()` 1.45s
cumulative (51%), IQ generation 0.81s (28%), detector 0.26s, jammer
0.19s -- the same shape as the C++ monolith's callgrind profile, shifted
up two orders of magnitude. The single biggest non-stage line item is
550,059 calls to protobuf's `RepeatedCompositeContainer.add` (one per
generated sample plus one per detected event).

**Microservice chain**, each service profiled during one run --
this is the pipelined-bottleneck structure measured directly:

```
    detector_service       1.12s working  (0.78s generation, 0.25s detection)
    spectrogram_service    1.41s working  (the bottleneck stage)
    jammer_service         0.20s working, 1.41s blocked in socket recv
    stats_service          0.03s working, 1.71s blocked in socket recv
    deinterleave_service   0.07s working, 1.80s blocked in socket recv
```

Downstream stages spend nearly their whole lives waiting on the
spectrogram stage -- the sink's steady-state number *is* the
spectrogram stage's throughput, which is exactly what "Two different
things are being measured" above predicted the sink metric would
degenerate to once one stage dominates.

**numpy variant** (2.08s under profiler): the vectorized kernels
themselves have nearly vanished from the profile -- what remains is IQ
generation (0.76s, still the interpreted sequential RNG) plus the
protobuf-to-array conversion (`np.fromiter` over three per-sample
generator expressions, ~0.55s cumulative). In other words, the
"conversion tax" described in "Going further" isn't a footnote, it *is*
this variant's profile: vectorizing the compute exposed
generation+extraction as the new dominant cost, the same
un-masking dynamic the multiproc/pickle finding showed at the
architecture level.

**numba variant** (0.86s total under profiler, including cache-load
warm-up excluded from the steady-state timer): the jitted kernels are
so cheap that the biggest remaining *steady-state* items are the two
deliberately un-jitted pure-Python stages -- the deinterleaver (59ms)
and stats accumulator (26ms) -- plus rebuilding `PulseEvent` protobuf
messages from the kernel's output arrays. The next optimization target
in this build wouldn't be a hot loop at all; it would be those
seams.

**multiproc's pickle overhead, isolated.** cProfile doesn't follow
`fork()`ed children, so the multiproc workers were characterized by
their computational identity to the microservice stages (same
`pulsecore` code) plus a targeted microbenchmark of the one thing that
differs -- the transport. Sending this repo's actual payload sizes 500
times through a `multiprocessing.Pipe` (which pickles every payload)
versus a raw framed TCP socket:

```
    265KB payload (the two iq-bearing hops):  pipe 304.6us/msg  socket 60.5us/msg   5.0x
     35KB payload (the two stripped hops):    pipe  16.3us/msg  socket 11.4us/msg   1.4x
```

`Connection.send()` pays a 5x per-message penalty on the big frames --
roughly 25ms per 50-batch run summed over the iq-bearing hops, the
right order of magnitude for the 1.08-1.10x multiproc-vs-microservices
gap the benchmarks show (and small enough to hide inside both builds'
run-to-run noise, which is why the two traded places before and after
the hyperoptimization pass).

## Profile-driven optimization, round 2

The profiling section above isn't just an explanation of the benchmark
numbers -- read as a to-do list, each finding names the next thing worth
attacking. This pass executed five candidates from it. Four survived;
one was measured, found to be a regression, and reverted -- documented
below with the same prominence as the wins, because *why* it lost is the
most instructive result in the pass. Every surviving change was
verified the usual way: all eight builds still byte-identical on every
numeric field at 1,000 and 50,000 pulses after each change and again
after a final clean rebuild, and both dedicated verification tools
(`scripts/verify_avx_variant.sh`,
`python/numpy_variant/verify_numpy_variant.py`) still PASS.

### The big one: protobuf's Clear() was deleting the chain's world every batch

Callgrind attributed ~45% of the iq-carrying C++ services' instructions
to `DestroyProtos` + `CreateMaybeMessage<IQSample>` + malloc/free --
destruction and re-creation of all 10,000 parsed `IQSample` objects,
every batch, in every consuming service. The root cause is visible in
the generated `pulse.pb.cc`: for a message *not* allocated on a
protobuf arena, the generated `PipelineFrame::Clear()` doesn't clear
singular submessage fields in place -- it **deletes them outright**
(`if (GetArenaForAllocation() == nullptr && _impl_.iq_ != nullptr)
delete _impl_.iq_;`). And `ParseFromString()` is Clear-then-merge, so
the "reuse one frame object across the loop" optimization from the
first pass never actually reused any submessage storage on the parse
side -- the frame object survived, its entire contents were rebuilt
from the heap up every batch.

The fix exploits the one protobuf container that *does* cache:
`RepeatedPtrField::Clear()` zeroes its elements and keeps them
allocated for the next `Add()`. So each consuming service now clears
the two big repeated-field carriers in place and parses with merge
semantics on top:

```cpp
if (frame.has_iq()) frame.mutable_iq()->Clear();
if (frame.has_events()) frame.mutable_events()->Clear();
if (!frame.MergeFromString(payload)) { ... }
```

Merging into a cleared element is value-identical to parsing into a
fresh one (Clear() zeroes every field; proto3 merge overwrites scalars
and appends to now-empty repeated fields), verified by output diff.
Effect, comparing ratios within a single benchmark run (which cancels
out machine drift -- see the caveat under the results table below):
**the C++ chain went from 2.97x the monolith to 1.97x**, its wall-clock
mean roughly halving. This one fix recovered more of the chain's
architectural overhead than everything in the first optimization pass
combined -- and it was invisible until callgrind said where the
instructions were going.

*Coda, caught by re-profiling:* the parse-side fix alone didn't finish
the job. Re-running callgrind afterward showed `jammer_service` still
carrying the full destruction churn -- because the generated
*field-level* accessor `clear_iq()`, which jammer calls before
forwarding, deletes the submessage exactly the way the message-level
`Clear()` does, so the delete had simply relocated from the parse to the
forward path. All the per-hop field clears are now in-place
`mutable_x()->Clear()` calls (cost: the forwarded frame carries a
2-byte present-but-empty field). That took `jammer_service` from 414M
to 169M instructions. Its *wall-clock* contribution barely moved --
jammer wasn't the chain's bottleneck stage, and a pipeline drains at
the slowest stage's rate regardless of how much slack the others have
-- which is itself the cleanest demonstration this repo has of the
bottleneck rule its "Two different things are being measured" section
describes: total chain CPU dropped ~20%, steady-state wall time didn't.

### The instructive failure: the C++ flat-array spectrogram, measured and reverted

Cachegrind's headline number -- 75.7% of the monolith's L1d read-misses
inside `SpectrogramAnalyzer::Process`, from re-walking 10,000
heap-scattered messages once per bin -- made the same fix Python's
spectrogram already had look mandatory in C++: extract i/q to flat
vectors once, then run the 8 bin passes over dense doubles. It was
implemented, verified bit-identical, and A/B measured:

```
    without flat extraction:  22.28ms mean   (15 runs)
    with flat extraction:     23.07ms mean   (15 runs)
```

A consistent ~3% *regression*, so it was reverted; pulsecore's scalar
spectrogram is unchanged. The explanation is already in the cachegrind
data, one line below the headline: the LL (last-level) miss rate is
~0%, meaning every one of those L1 misses hits in L2 -- a ~12-cycle
penalty that out-of-order execution overlaps almost entirely with the
correlator's real arithmetic. The misses were *counted*, but they were
never *stalling* anything, and the extraction pass added genuine new
work (20,000 stores plus 160KB of traffic per batch) to fix a
non-problem. **Cachegrind counts misses, not stalls** -- the same
lesson as the AVX pass, one level deeper: before optimizing a
memory-system number, check whether the memory system is actually the
thing holding the retirement rate back. (The identical fix pays hugely
in Python -- see below -- because CPython's cost is interpreter
dispatch per access, not latency per access.)

### Python: the spectrogram inner loop is complex multiplication -- so let complex do it

cProfile put the spectrogram at 51% of the Python monolith's runtime.
Its inner loop -- four multiplies and two adds to correlate, four
multiplies and two adds to rotate -- is exactly two complex
multiplications, and CPython evaluates a `complex * complex` in C using
the same component formulas, same operations, same order (verified
bit-identical over 100,000 random cases before adopting, and the
magnitude deliberately stays `sqrt(re*re + im*im)` rather than
`abs()`, which goes through `hypot()` and rounds differently). Samples
are read out of protobuf once per batch into a `complex` list, and the
inner loop becomes `acc += s * rot; rot *= step` -- roughly half the
interpreted bytecode per sample. Together with a smaller fix in the
generator (one `add(**kwargs)` call per sample instead of `add()` plus
three attribute assignments -- 30,000 fewer interpreted operations per
batch), the Python monolith went from 1619ms to 1336ms back-to-back
(~1.2x), and every Python build sharing `pulsecore/` (microservices,
multiproc) inherits both fixes.

### numpy: stop generating through protobuf

The profiling section measured this variant's generation-plus-extraction
path as bigger than all its vectorized kernels combined. The fix is the
same design numba_variant already used: generate straight into numpy
arrays (`numpy_variant/iq_source_arrays.py`) -- indices and the
pulse/gap envelope fully vectorized, only the sequential xorshift
recurrence left in a Python loop, and nothing routed through
`pulse_pb2.IQBatch` at all. Bit-identity with the protobuf generator is
enforced by a new check in `verify_numpy_variant.py`, not assumed.
Back-to-back: 1219ms to 693ms (~1.76x); the variant now runs at half
the Python monolith's time instead of three-quarters.

### numba: the seams were the bottleneck, so the seams got compiled

The profiling section ended by noting the numba build's remaining
steady-state time wasn't in any hot loop -- it was the two deliberately
un-jitted pure-Python stages (deinterleaver, stats) plus rebuilding
~1,000 `PulseEvent` protobuf messages per batch solely to feed them.
This pass jitted both (`stats_accumulate`, `deinterleave_events` in
`numba_variant/kernels.py` -- same event order, same operations, so
byte-identical output) and deleted the protobuf rebuild, which nothing
needed anymore. Back-to-back: 143ms median to 30ms (~4.3x). All six
pieces of per-batch work are now compiled, and nothing in the hot path
touches protobuf.

### Results after round 2

Same 8-way benchmark (`./scripts/benchmark_steady_state.sh 20 50000`):

```
Steady-state results over 20 runs, 50000 pulses/run:

            C++ monolith  min=   17.418ms  median=   17.644ms  mean=   17.769ms  stdev=   0.382ms  max=   18.937ms
     C++ monolith (AVX2)  min=   17.242ms  median=   17.635ms  mean=   17.705ms  stdev=   0.384ms  max=   18.707ms
       C++ microservices  min=   32.203ms  median=   34.398ms  mean=   34.995ms  stdev=   1.909ms  max=   39.289ms
         Python monolith  min= 1079.874ms  median= 1128.165ms  mean= 1125.021ms  stdev=  16.758ms  max= 1148.538ms
    Python microservices  min=  641.209ms  median=  655.288ms  mean=  663.368ms  stdev=  18.620ms  max=  704.197ms
        Python multiproc  min=  658.602ms  median=  695.283ms  mean=  696.056ms  stdev=  22.682ms  max=  739.024ms
            Python numba  min=   29.411ms  median=   30.357ms  mean=   30.662ms  stdev=   1.442ms  max=   35.732ms
            Python numpy  min=  542.651ms  median=  562.800ms  mean=  567.255ms  stdev=  14.196ms  max=  595.905ms

Python monolith is 63.3x the C++ monolith mean
Python microservices is 19.0x the C++ microservices mean
C++ microservices is 1.97x the C++ monolith mean
Python microservices is 0.59x the Python monolith mean
Python multiproc is 0.62x the Python monolith mean
Python multiproc is 1.05x the Python microservices mean
C++ AVX2 monolith is 1.00x the C++ (scalar) monolith mean
Python numba is 0.027x the Python monolith mean (36.7x faster)
Python numpy is 0.50x the Python monolith mean (1.98x faster)
```

A caveat on cross-run comparisons: the C++ monolith -- whose code this
pass didn't change -- has measured anywhere from ~18ms to ~23ms mean
across different benchmark sessions on this machine, so *absolute*
before/after deltas between runs carry that drift. The ratios within a
single run don't, which is why the chain improvement is stated as
2.97x-to-1.97x rather than as milliseconds; the back-to-back deltas
quoted per-change above were measured minutes apart specifically to
sidestep this.

Two standout shifts in the final ordering: **Python-with-numba now
beats the C++ microservice chain outright** (30.7ms vs 35.0ms mean) and
sits within 1.7x of the C++ monolith itself -- "which language" turns
out to matter far less than "how many times does the same batch get
serialized, parsed, and re-heap-allocated." And the C++ chain's
overhead over its monolith, the number this whole repo exists to
examine, is now below 2x -- with the remaining gap being genuine
serialize/parse/syscall work that a process boundary actually requires,
rather than an accidental delete-and-rebuild the profiler had to
catch.

### Profiling, re-run after round 2

The same instrumentation that motivated round 2, re-run against the
finished tree to confirm each fix landed where its profile said it
would (and to catch the one place it hadn't -- the `clear_iq()` coda
above came from exactly this re-run):

**C++ chain, per-service callgrind totals** (50,000 pulses; "before" is
the original profiling section above):

```
                          before      after       what changed
    detector_service       151M Ir     151M Ir    nothing (producer never parses)
    spectrogram_service    601M Ir     336M Ir    merge-parse: iq tree reused
    jammer_service         436M Ir     169M Ir    merge-parse + in-place field clears
    stats_service           57M Ir      33M Ir    merge-parse: events reused
    deinterleave_service    47M Ir      23M Ir    merge-parse: events reused
    ----------------------------------------
    chain total          1,292M Ir     712M Ir    7.7x the monolith's 167M -> 4.3x
```

The destruction churn is gone from every consumer -- `DestroyProtos`/
`CreateMaybeMessage<IQSample>` fell from ~28% of `jammer_service` to
~1% (the one-time first-batch allocation and final teardown, which is
what those numbers *should* be).

**Python builds, cProfile** (50,000 pulses, times under profiler
overhead -- compare shapes, not absolutes):

- *Monolith* (2.85s -> 1.62s profiled): the spectrogram fell from 51%
  of runtime to 29%; IQ generation is now the largest single item.
  There is no longer one dominant stage -- the profile is flat-ish,
  which is where diminishing returns set in for this build.
- *numpy* (2.08s -> 0.93s): the protobuf-to-array extraction is gone
  from the profile entirely; the largest remaining item is the
  sequential RNG loop in `iq_source_arrays.next_batch` (0.33s) -- the
  one thing this variant documented as not vectorizable without
  jump-ahead machinery. The variant is now bounded by exactly the
  limitation its docstrings claimed. (This finding didn't stay final:
  the jump-ahead machinery got built after all -- see "Vectorizing the
  'unvectorizable' RNG" below.)
- *numba* (0.66s profiled, nearly all of it JIT cache loading outside
  the steady-state timer): the hot path has effectively vanished from
  the Python-level profile -- the largest genuine work item is the
  spectrogram kernel at 13ms. There is nothing interpreted left to
  optimize; further gains would have to come from the kernels
  themselves.

### Vectorizing the "unvectorizable" RNG (GF(2) jump-ahead)

Round 2 left numpy_variant bounded by its sequential RNG loop and the
docs calling jump-ahead "out of proportion." With the loop now the
variant's *only* dominant cost, the proportion argument flipped, so the
machinery got built (`numpy_variant/iq_source_arrays.py`).

The insight that makes it possible: every operation in xorshift32's
step (`s ^= s<<13; s ^= s>>17; s ^= s<<5`) is an XOR of bit-shifts, so
one step is a linear map over GF(2) -- a 32x32 bit-matrix `M` with
`state(n+k) = M^k * state(n)`, and `M^k` computable once by
square-and-multiply. "Inherently sequential" turns out to mean
"sequential only if you evaluate it that way":

- A full batch's 20,000 draws split into 2,500 lanes of 8 consecutive
  draws each; lanes own *contiguous* chunks, so a row-major grid
  flattens into exactly the scalar sequence order.
- Within a batch, all lanes advance together: the plain three-shift
  step applied elementwise to a uint32 array, 8 times (~150 numpy calls
  replacing 20,000 interpreted iterations).
- Between batches, every lane jumps the same fixed distance via one
  precomputed `M^(20000-8)` applied vectorized.
- Lane seeding is one scalar pre-run at construction time, outside the
  steady-state timer under the same rule that excludes numba's JIT
  warm-up; the 8-sample tail batch falls back to the scalar loop.

Everything is exact integer XOR/shift until the final elementwise
float conversion, so the output is **bit-identical** to the scalar
generator -- same sequence, different evaluation order -- and that's
enforced by the existing generator-equality checks in
`verify_numpy_variant.py` and the unit tests, which compare against the
protobuf generator sample for sample and don't care how the numbers
were produced.

Measured effect: the microbenchmarked RNG went from ~6.5ms to ~0.4ms
per batch (16x), and the variant's steady state from ~565ms to ~462ms
median back-to-back (~1.2x). That's real but smaller than the ~330ms
the round-2 cProfile attributed to the loop -- profiler overhead
inflates per-iteration loop costs, a measurement caveat worth naming.
Post-change profiling shows the RNG gone from the profile entirely; the
variant's floor moved to per-*event* work -- which the next round then
attacked (below). The bottleneck has now moved four times in this one
variant, each move measured, which is about as clean an illustration of
iterative profiling as this repo has to offer.

### The next batch: reduceat events, cached phase tables, prange bins

One more reprofile-then-fix cycle, three changes, all
verification-gated:

1. **`detect_pulses` fully vectorized via `np.ufunc.reduceat`** (numpy
   variant). The per-event Python loop made two tiny-array reductions
   per detected pulse -- ~100k numpy calls per run, the kernel's
   dominant cost. Interleaving the rising/falling edge indices gives
   reduceat boundaries where every even-indexed segment is a pulse:
   two reduceat calls now aggregate *every* in-batch pulse's peak and
   sum. Exactness was validated empirically before adopting, and the
   finding is worth recording: `np.maximum.reduceat` reduces strictly
   sequentially (peaks stay exact for any input), but `np.add.reduceat`
   reassociates sums from segment length 3 -- and the *previous* code's
   `seg.sum()` already reassociated from length 8, so the detector's
   "exact" claim was always silently conditional on short pulses. The
   docstring now states the real contract: everything exact except
   per-event `mean_amplitude` for pulses of 3+ samples -- a field
   nothing downstream reads, invisible in every build's output, and
   exact anyway for this repo's 2-sample pulses (the verify tool's
   exact event comparison still passes unchanged).
2. **Cached per-bin phase tables** (numpy variant). The spectrogram
   recomputed `cos`/`sin` over 80,000 points per batch; the absolute
   phase `omega*(first+k)` factors into a per-batch scalar phasor times
   a batch-shape-constant offset table, so the transcendentals now
   happen once per batch *shape* (two shapes exist: 10,000 and 8).
   One extra rounding step per element (a complex multiply), and the
   measured error vs. the scalar reference actually *improved* -- from
   3.0e-11 to 1.5e-12 worst-case -- because small-argument cos/sin
   avoids the precision the direct evaluation loses reducing arguments
   like `omega*500000`.
3. **`prange` over spectrogram bins** (numba variant). The 8 bins are
   fully independent correlators, so threading them (numba
   `parallel=True`) changes no bin's operation order -- bit-identity
   enforced by the existing parity tests. 31.6ms -> 24.2ms median
   back-to-back.

Net, back-to-back: **numpy 462ms -> 192ms median** (2.4x this round;
within-run, 7.8x faster than the Python monolith, where round 2 left it
at 2.0x), **numba 31.6ms -> 24.2ms** -- now faster than everything in
the repo except the two C++ monoliths, C++ chain included. All 42 unit
tests, both verify tools, and the cross-build output diffs pass
unchanged; the fresh 8-way run confirming these standings is in the
git history alongside this change (absolute numbers that session ran
~15% slow machine-wide -- the drift caveat above -- which is why the
per-change numbers here are back-to-back pairs and within-run ratios).

## Tests

`./scripts/run_tests.sh` builds and runs both unit-test suites:

- **C++** (`tests/pulsecore_tests.cpp`, run via `ctest`): every
  pulsecore algorithm, the TCP framing layer (round-trip, empty
  payload, >1MB partial-read/write paths, peer-close semantics), and --
  when the AVX2 variant was built -- exact-match parity for the AVX2
  detector and tolerance parity for the AVX2 jammer. Framework-free on
  purpose: a CHECK macro and a failure count keep the repo's dependency
  footprint at zero.
- **Python** (`python/tests/`, stdlib `unittest`): mirrors the C++
  suite case for case, plus kernel-parity tests for the numba/numpy
  variants that *skip* (not fail) when those optional dependencies
  aren't installed.

Three details worth knowing about the suites' design:

- **The golden values are shared.** Both suites assert the same pinned
  doubles for the synthetic generator's output (sample 0, 7, 8, and
  107 of a seed-42 run). The repo's cross-language bit-identity claim
  stops being something verified occasionally by output diff and
  becomes something every test run enforces in both languages
  independently.
- **They cover what the synthetic signal can't.** At this repo's
  constants the batch size is an exact multiple of the pulse period,
  so a pulse never straddles a batch boundary in any benchmark run --
  but the detector state machine (in all four implementations: scalar
  C++, AVX2, pure Python, and both variant kernels) explicitly supports
  it, and only these tests exercise it.
- **One test documents a limitation instead of hiding it.** The
  two-emitter deinterleaver test constructs its input so the first
  emitter's PRI is established before the second appears, with a
  comment explaining why: two trains interleaved from a cold start get
  their seed pulses merged -- an inherent property of the simplified
  sequential-PRI algorithm. The first draft of that test assumed
  otherwise, failed, and the failure was the algorithm telling the
  truth about itself; the test now pins the documented behavior.

## Correctness

Every change in this repo's history was re-verified the same way: run the
relevant binaries/scripts at a small pulse count and a larger one and diff
the printed output. All five builds -- C++ monolith, C++ microservices,
Python monolith, Python microservices, Python multiproc -- have produced
byte-identical summary statistics, spectrogram bins, jam-detection state,
and deinterleaved tracks in every case, including after adding the three
new apps, after the spectrogram phasor-rotation fix, after converting the
fan-out topology into a linear chain, after reordering that chain and
clearing unused fields between hops, after scaling the signal rate up to
1,000,000 pulses/sec, after adding the Python port, after adding the
`STEADY_STATE_MS` instrumentation for the steady-state benchmark above
(every program's non-timing output is unchanged; the new timer variables
only wrap existing loops and add one new printed line each), and after
adding the Python multiproc build (verified against both the Python
monolith's and the C++ monolith's output at 1000 and 50,000 pulses, and
re-run five times back to back to confirm the forked workers' explicit
`None`-sentinel shutdown -- see "A third Python architecture" above --
never leaves a process hung or a result nondeterministic), and after the
"Correctness audit and hyperoptimization pass" above (diffed against a
full clean rebuild and proto regeneration, all five builds still
byte-identical on every field after every optimization -- see that
section for specifics), and after adding the AVX2/numba/numpy variants
in "Going further" above -- eight builds now checked against a full
clean rebuild and proto regeneration, with numba_variant and the AVX2
detector held to the same exact byte-for-byte bar as the original five
(see `scripts/verify_avx_variant.sh`), and numpy_variant plus the AVX2
jammer held to a measured, documented floating-point tolerance instead,
since exact equality isn't achievable for a vectorized reduction and
claiming otherwise would be dishonest (see
`python/numpy_variant/verify_numpy_variant.py`), and after the second,
profile-driven optimization pass ("Profile-driven optimization,
round 2" above): the C++ merge-parse change, the Python complex-
arithmetic spectrogram, the numpy array-native generator, and the fully
jitted numba build were each diffed at both scales as they landed, the
reverted C++ flat-array spectrogram was verified bit-identical *before*
being rejected on performance grounds (correct-but-slower is still
reverted), and both verification tools pass against the final tree.
Clearing a field a
stage already consumed and copying its value into a local variable first
(see `spectrogram_service`/`jammer_service`/`stats_service`'s
`last_summary` pattern, in both languages) is exactly the kind of change
that's easy to get backwards -- clear-then-read instead of read-then-clear
silently zeroes out the value you meant to print -- so this one got the
same before/after diff treatment as everything else. The scale-up
specifically was checked against `PulseSummary.mean_pri_seconds` coming
out to exactly `0.000001` (confirming the 1,000,000-pulse/sec rate) and
the deinterleaver's single track's `pulse_count`/`estimated_pri_seconds`
matching `stats`' exactly, at both 1000 pulses (one buffer) and
1,000,000 pulses (one full second) -- in all four builds. The Python port
additionally relies on `pulsecore/iq_source.py`'s xorshift32 RNG matching
the C++ version bit-for-bit (verified by diffing full output, not just
summary statistics, since a subtly different RNG would still pass the
"detected 1000000 pulses at the right rate" checks while generating
different noise).
