// jammer_service: standalone executable running the exact same
// pulsecore::JammerDetector used by libpulse_jammer_plugin.so in the
// monolith build. Third stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver): connects out to
// stats_service (the next stage) at startup, then listens for
// spectrogram_service.
//
// This is the last of the three stages that read frame.iq() (the
// largest field by far -- 10,000 samples/batch), which is exactly why
// detector, spectrogram, and jammer are grouped first in the chain: once
// this stage is done with it, nothing downstream (stats_service,
// deinterleave_service) ever reads it again, so it's cleared right here
// before forwarding instead of being serialized and transmitted two more
// times for no reason.

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "jammer.h"
#include "pulse.pb.h"

int main(int argc, char** argv) {
  const uint16_t listen_port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50053;
  const std::string next_host = argc > 2 ? argv[2] : "127.0.0.1";
  const uint16_t next_port = argc > 3 ? static_cast<uint16_t>(std::atoi(argv[3])) : 50054;
  constexpr double kPowerThreshold = 20.0;
  constexpr double kDutyCycleThreshold = 0.5;

  std::printf("[jammer_service] connecting to stats_service at %s:%u\n", next_host.c_str(),
              next_port);
  const int downstream_fd = netutil::Connect(next_host, next_port);
  if (downstream_fd < 0) {
    std::fprintf(stderr, "[jammer_service] failed to connect (is stats_service running?)\n");
    return 1;
  }

  const int listen_fd = netutil::Listen(listen_port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[jammer_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf(
      "[jammer_service] listening on 127.0.0.1:%u, waiting for spectrogram_service...\n",
      listen_port);

  const int upstream_fd = netutil::Accept(listen_fd);
  if (upstream_fd < 0) {
    std::fprintf(stderr, "[jammer_service] accept failed\n");
    return 1;
  }
  std::printf("[jammer_service] spectrogram_service connected\n");

  pulsecore::JammerDetector detector(kPowerThreshold, kDutyCycleThreshold);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PipelineFrame frame;
  // Kept separately from frame because frame.jam() gets cleared before
  // every forward (see below) -- this is what gets printed after the
  // loop ends.
  pulse::JamSummary last_summary;
  int frames_forwarded = 0;
  // See spectrogram_service/main.cpp for why the timer starts on the
  // first successful receive rather than before the loop.
  std::chrono::steady_clock::time_point steady_state_start;
  std::chrono::steady_clock::time_point steady_state_end;
  bool started = false;
  while (netutil::RecvMessage(upstream_fd, &payload)) {
    if (!started) {
      steady_state_start = std::chrono::steady_clock::now();
      started = true;
    }
    // In-place clear + merge-parse instead of ParseFromString(), so the
    // 10,000 parsed IQSample objects get reused across batches instead
    // of destroyed and re-allocated by the non-merge parse's implicit
    // Clear() -- see spectrogram_service/main.cpp for the full story
    // (callgrind attributed ~45% of this process's instructions to that
    // churn).
    if (frame.has_iq()) frame.mutable_iq()->Clear();
    if (frame.has_events()) frame.mutable_events()->Clear();
    if (!frame.MergeFromString(payload)) {
      std::fprintf(stderr, "[jammer_service] dropping malformed frame\n");
      continue;
    }
    detector.Process(frame.iq(), frame.mutable_jam());
    last_summary = frame.jam();

    // Nothing downstream (stats_service, deinterleave_service) reads
    // frame.iq() or frame.jam() -- this is the last stage that needs the
    // raw samples, so drop them here instead of paying to move the
    // biggest message in the pipeline across two more hops unused.
    frame.clear_iq();
    frame.clear_jam();

    frame.SerializeToString(&payload);
    if (!netutil::SendMessage(downstream_fd, payload)) {
      std::fprintf(stderr, "[jammer_service] forward failed, stats_service may have exited\n");
      break;
    }
    ++frames_forwarded;
  }
  steady_state_end = std::chrono::steady_clock::now();
  const double steady_state_ms =
      started
          ? std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count()
          : 0.0;

  std::printf("[jammer_service] received/forwarded %d frame(s) over TCP\n", frames_forwarded);
  std::printf("[jammer_service] STEADY_STATE_MS %.6f\n", steady_state_ms);
  std::printf(
      "[jammer_service] %llu/%llu batches flagged, max_duty_cycle=%.3f max_mean_power=%.2f\n",
      static_cast<unsigned long long>(last_summary.batches_flagged()),
      static_cast<unsigned long long>(last_summary.batches_total()), last_summary.max_duty_cycle(),
      last_summary.max_mean_power());

  ::close(upstream_fd);
  ::close(downstream_fd);
  ::close(listen_fd);
  return 0;
}
