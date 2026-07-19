#pragma once

// Theoretical-limits branch: a stage-per-thread pipeline harness for the
// two monolith apps. The main branch kept the monolith single-threaded
// on purpose -- threading its stages would turn it into the very thing
// it's being compared against -- and the max-optimization branch upheld
// that. This branch's charter is speed, and the shm-ring round already
// showed the five *processes* overlapping their stages once transport
// got cheap (the chain's min dipped below the monolith's). This header
// gives the monolith the same pipeline parallelism the chain gets for
// free from being five processes, minus everything the chain still
// pays: there is no serialization, no parse, no copy, and no ring --
// stages hand each other a pulse::PipelineFrame pointer and the only
// synchronization is one release-store/acquire-load pair per stage per
// batch.
//
// Mechanics: a small fixed ring of pre-allocated PipelineFrame "slots"
// (protobuf object reuse warms up exactly like the single-threaded
// loop's one reused frame did -- each slot's repeated fields reach
// steady-state capacity after its first few laps). Batch n lives in
// slot n % kSlots. Stage s may process batch n once stage s-1 has
// published done[s-1] >= n+1; the producer (generation fused with the
// first stage, mirroring detector_service's role in the chain) may
// refill a slot once the LAST stage has drained the batch that used it
// kSlots ago. Every stage sees every batch, in order -- required, since
// detector/stats/deinterleaver all carry running state across batches
// (a batch-parallel design would change results; this one reorders
// nothing and keeps per-stage arithmetic identical to the sequential
// loop's).
//
// Waits spin briefly then sched_yield(), same policy (and reasoning) as
// microservice/net/framing.cpp: more runnable threads than cores here,
// so hogging a core while blocked would slow the very stage being
// waited on.

#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "iq_source.h"
#include "pulse.pb.h"

namespace monolith {

struct ThreadedPipelineResult {
  int batches = 0;
  double steady_state_ms = 0.0;
  // Frame the final batch flowed through: every stage wrote its
  // last-batch summary into this one, so it's the frame to print.
  // Points into the caller's slot vector (or slot 0 if no batches ran).
  pulse::PipelineFrame* final_frame = nullptr;
};

namespace internal {
inline void WaitPause(int& spins) {
  if (++spins < 256) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  } else {
    ::sched_yield();
  }
}
// One counter per cache line -- five threads spin-read their upstream
// neighbor's counter, so sharing a line would ping-pong it.
struct alignas(64) PaddedCounter {
  std::atomic<uint64_t> value{0};
};
}  // namespace internal

// Runs source -> stages[0] (fused, the producer thread) -> stages[1] ->
// ... -> stages.back(), one thread per stage entry, over the caller's
// slot ring. The caller owns `slots` (so final_frame stays valid after
// return) and chooses the stage grouping -- fewer entries = fewer
// threads; group cheap stages together to match the core count.
// steady_state_ms covers thread spawn to last join, the same
// batch-loop-only span the single-threaded version timed (thread
// create/join is microseconds against a multi-ms pipeline).
inline ThreadedPipelineResult RunThreadedPipeline(
    pulsecore::SyntheticIQSource* source,
    const std::vector<std::function<void(pulse::PipelineFrame*)>>& stages,
    std::vector<pulse::PipelineFrame>* slots) {
  const uint64_t num_slots = slots->size();
  const size_t num_stages = stages.size();
  std::vector<internal::PaddedCounter> done(num_stages);
  std::atomic<int64_t> total{-1};  // batch count, published by the producer at end-of-stream

  const auto steady_state_start = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(num_stages);

  // Producer: generation fused with the first stage (the chain fuses
  // them into detector_service for the same reason -- generation feeds
  // nothing but the first stage, so a seam there buys only overhead).
  threads.emplace_back([&] {
    uint64_t n = 0;
    for (;;) {
      if (n >= num_slots) {
        // Batch n reuses batch (n - num_slots)'s slot; wait for the
        // last stage to have fully drained it.
        int spins = 0;
        while (done[num_stages - 1].value.load(std::memory_order_acquire) <
               n - num_slots + 1) {
          internal::WaitPause(spins);
        }
      }
      pulse::PipelineFrame* frame = &(*slots)[n % num_slots];
      if (!source->NextBatch(frame->mutable_iq())) break;
      stages[0](frame);
      done[0].value.store(n + 1, std::memory_order_release);
      ++n;
    }
    total.store(static_cast<int64_t>(n), std::memory_order_release);
  });

  for (size_t s = 1; s < num_stages; ++s) {
    threads.emplace_back([&, s] {
      uint64_t n = 0;
      for (;;) {
        int spins = 0;
        while (done[s - 1].value.load(std::memory_order_acquire) < n + 1) {
          const int64_t t = total.load(std::memory_order_acquire);
          if (t >= 0 && static_cast<int64_t>(n) >= t) return;  // drained: end of stream
          internal::WaitPause(spins);
        }
        stages[s](&(*slots)[n % num_slots]);
        done[s].value.store(n + 1, std::memory_order_release);
        ++n;
      }
    });
  }

  for (std::thread& t : threads) t.join();

  const auto steady_state_end = std::chrono::steady_clock::now();

  ThreadedPipelineResult result;
  const int64_t batches = total.load(std::memory_order_relaxed);
  result.batches = static_cast<int>(batches);
  result.steady_state_ms =
      std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count();
  result.final_frame =
      batches > 0 ? &(*slots)[static_cast<uint64_t>(batches - 1) % num_slots] : &(*slots)[0];
  return result;
}

}  // namespace monolith
