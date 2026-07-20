#pragma once

// Theoretical-limits branch, round three: a stage-per-thread pipeline
// harness for the two monolith apps, reworked for the branch's revised
// benchmark charter -- **the signal is a given**. In the real system
// this repo caricatures, IQ samples arrive from a radio; no
// architecture choice makes the antenna faster. So generation is no
// longer a pipeline stage at all: the caller pre-generates every
// batch's frame before calling RunThreadedPipeline, and the measured
// region starts at the first stage that is actually this system's job
// -- detection.
//
// That charter change deleted this harness's previous complexity. With
// one pre-filled frame per batch there is no slot ring, no slot-reuse
// gating, and no producer thread fused with generation: every stage is
// uniform. Thread s processes frames strictly in order, waiting only
// for stage s-1 to have published each frame (stage 0 waits for
// nothing -- its input already exists). The only synchronization is
// still one release-store/acquire-load pair per stage per batch, on
// counters padded to a cache line each.
//
// Stages must still see every batch in order from one thread --
// detector/stats/deinterleaver carry running state across batches --
// and that invariant is what this harness provides. Waits spin briefly
// then sched_yield(), same policy (and reasoning) as
// microservice/net/framing.cpp.
//
// Deliberately NOT pinned to cores, and that's a measured result, not
// an omission: pthread_setaffinity_np per stage thread was tried and
// LOST (median 6.2ms pinned vs 4.7ms unpinned, 10 runs each) -- this
// repo's benchmark box is a shared container, and pinning traps a
// stage on a core the neighbors happen to be loading instead of
// letting the scheduler migrate it somewhere idle. Pinning pays on
// isolated hardware; this is not that.

#include <sched.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "pulse.pb.h"

namespace monolith {

struct ThreadedPipelineResult {
  int batches = 0;
  double steady_state_ms = 0.0;
  // Frame the final batch flowed through: every stage wrote its
  // last-batch summary into it, so it's the frame to print. Points
  // into the caller's frame vector (frames->back(), or nullptr when
  // the input was empty).
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
// One counter per cache line -- each thread spin-reads its upstream
// neighbor's counter, so sharing a line would ping-pong it.
struct alignas(64) PaddedCounter {
  std::atomic<uint64_t> value{0};
};
}  // namespace internal

// Runs stages[0] -> stages[1] -> ... -> stages.back() over the
// caller's pre-generated frames, one thread per stage entry. The
// caller owns `frames` (so final_frame stays valid after return),
// pre-fills each frame's iq before calling, and chooses the stage
// grouping -- fewer entries = fewer threads; group cheap stages
// together to match the core count. steady_state_ms covers thread
// spawn to last join: detection through deinterleave, with generation
// finished before the clock starts.
inline ThreadedPipelineResult RunThreadedPipeline(
    const std::vector<std::function<void(pulse::PipelineFrame*)>>& stages,
    std::vector<pulse::PipelineFrame>* frames) {
  const uint64_t num_frames = frames->size();
  const size_t num_stages = stages.size();
  std::vector<internal::PaddedCounter> done(num_stages);

  // Backpressure window: stage 0 may run at most this many frames
  // ahead of the last stage. Without it the fastest stage sprints
  // through the whole pre-generated input and every stage behind it
  // reads stone-cold memory -- measured as a ~3x pipeline slowdown
  // when this cap was briefly absent (the round-two slot ring provided
  // it implicitly; this preserves the same cache-sized working set,
  // 8 x ~160KB, without the ring's frame reuse).
  constexpr uint64_t kWindow = 8;

  const auto steady_state_start = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  threads.reserve(num_stages);
  for (size_t s = 0; s < num_stages; ++s) {
    threads.emplace_back([&, s] {
      for (uint64_t n = 0; n < num_frames; ++n) {
        int spins = 0;
        if (s > 0) {
          while (done[s - 1].value.load(std::memory_order_acquire) < n + 1) {
            internal::WaitPause(spins);
          }
        } else if (n >= kWindow) {
          while (done[num_stages - 1].value.load(std::memory_order_acquire) + kWindow < n + 1) {
            internal::WaitPause(spins);
          }
        }
        stages[s](&(*frames)[n]);
        done[s].value.store(n + 1, std::memory_order_release);
      }
    });
  }
  for (std::thread& t : threads) t.join();

  const auto steady_state_end = std::chrono::steady_clock::now();

  ThreadedPipelineResult result;
  result.batches = static_cast<int>(num_frames);
  result.steady_state_ms =
      std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count();
  result.final_frame = num_frames > 0 ? &frames->back() : nullptr;
  return result;
}

}  // namespace monolith
