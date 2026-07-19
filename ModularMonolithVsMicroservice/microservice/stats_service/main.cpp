// stats_service: standalone executable running the exact same
// pulsecore::PulseStatsAccumulator used by libpulse_stats_plugin.so in the
// monolith build. Fourth stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver): connects out to
// deinterleave_service (the next and final stage) at startup, then
// listens for jammer_service. By this point frame.iq() has already been
// cleared upstream (see jammer_service/main.cpp) since nothing from here
// on needs it. For each frame: folds frame.events() into running stats,
// and forwards the frame downstream with its own stats field cleared
// before sending -- deinterleave_service doesn't read it.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "pulse.pb.h"
#include "pulse_stats.h"

int main(int argc, char** argv) {
  const uint16_t listen_port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50054;
  const std::string next_host = argc > 2 ? argv[2] : "127.0.0.1";
  const uint16_t next_port = argc > 3 ? static_cast<uint16_t>(std::atoi(argv[3])) : 50055;
  constexpr double kSampleRateHz = 10000000.0;

  std::printf("[stats_service] connecting to deinterleave_service at %s:%u\n", next_host.c_str(),
              next_port);
  netutil::Channel* downstream = netutil::Connect(next_host, next_port);
  if (downstream == nullptr) {
    std::fprintf(stderr,
                  "[stats_service] failed to connect (is deinterleave_service running?)\n");
    return 1;
  }

  netutil::Channel* listener = netutil::Listen(listen_port);
  if (listener == nullptr) {
    std::fprintf(stderr, "[stats_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf("[stats_service] listening on shm ring %u, waiting for jammer_service...\n",
              listen_port);

  netutil::Channel* upstream = netutil::Accept(listener);
  if (upstream == nullptr) {
    std::fprintf(stderr, "[stats_service] accept failed\n");
    return 1;
  }
  std::printf("[stats_service] jammer_service connected\n");

  pulsecore::PulseStatsAccumulator accumulator(kSampleRateHz);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PipelineFrame frame;
  // Kept separately from frame because frame.stats() gets cleared before
  // every forward (see below) -- this is what gets printed after the
  // loop ends.
  pulse::PulseSummary last_summary;
  int frames_forwarded = 0;
  // See spectrogram_service/main.cpp for why the timer starts on the
  // first successful receive rather than before the loop.
  std::chrono::steady_clock::time_point steady_state_start;
  std::chrono::steady_clock::time_point steady_state_end;
  bool started = false;
  while (netutil::RecvMessage(upstream, &payload)) {
    if (!started) {
      steady_state_start = std::chrono::steady_clock::now();
      started = true;
    }
    // In-place clear + merge-parse instead of ParseFromString(), so the
    // parsed PulseEvent objects (~1,000/batch -- iq was already stripped
    // upstream) get reused across batches instead of destroyed and
    // re-allocated by the non-merge parse's implicit Clear() -- see
    // spectrogram_service/main.cpp for the full story.
    if (frame.has_iq()) frame.mutable_iq()->Clear();
    if (frame.has_events()) frame.mutable_events()->Clear();
    if (!frame.MergeFromString(payload)) {
      std::fprintf(stderr, "[stats_service] dropping malformed frame\n");
      continue;
    }
    accumulator.Add(frame.events());
    *frame.mutable_stats() = accumulator.Finalize();
    last_summary = frame.stats();

    // deinterleave_service (next hop) only reads frame.events(); nothing
    // downstream of it ever reads frame.stats(). In-place Clear() rather
    // than clear_stats(), which would delete/re-allocate the submessage
    // every batch -- see jammer_service/main.cpp.
    frame.mutable_stats()->Clear();

    frame.SerializeToString(&payload);
    if (!netutil::SendMessage(downstream, payload)) {
      std::fprintf(stderr,
                    "[stats_service] forward failed, deinterleave_service may have exited\n");
      break;
    }
    ++frames_forwarded;
  }
  steady_state_end = std::chrono::steady_clock::now();
  const double steady_state_ms =
      started
          ? std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count()
          : 0.0;

  std::printf("[stats_service] received/forwarded %d frame(s) via shm ring\n", frames_forwarded);
  std::printf("[stats_service] STEADY_STATE_MS %.6f\n", steady_state_ms);
  std::printf(
      "[stats_service] pulses=%llu mean_peak=%.3f mean_dur_us=%.2f mean_pri_us=%.2f "
      "min_peak=%.3f max_peak=%.3f\n",
      static_cast<unsigned long long>(last_summary.pulse_count()), last_summary.mean_peak_amplitude(),
      last_summary.mean_duration_seconds() * 1e6, last_summary.mean_pri_seconds() * 1e6,
      last_summary.min_peak_amplitude(), last_summary.max_peak_amplitude());

  netutil::Close(upstream);  // == listener; consumer side unlinks the ring
  netutil::Close(downstream);
  return 0;
}
