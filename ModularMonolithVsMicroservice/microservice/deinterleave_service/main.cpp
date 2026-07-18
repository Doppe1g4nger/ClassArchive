// deinterleave_service: standalone executable running the exact same
// pulsecore::Deinterleaver used by libpulse_deinterleaver_plugin.so in the
// monolith build. Fifth and final stage of the pipeline chain (detector
// -> spectrogram -> jammer -> stats -> deinterleaver): it's the chain's
// sink, so it only listens for stats_service and never forwards -- it
// folds frame.events() into candidate emitter tracks and prints the
// result once the upstream connection closes.
//
// As the chain's sink, this process is also where the pipeline's overall
// steady-state measurement is taken: the time from its first successful
// receive to its last is, by construction, the time it took the whole
// pipeline to drain once fully connected, with zero cross-process
// timestamp correlation required. That works because no data can reach
// this process until every upstream hop (stats -> jammer -> spectrogram
// -> detector) has finished connecting -- see detector_service/main.cpp,
// whose own connect() can't succeed any earlier than that either. See
// scripts/benchmark_steady_state.sh for how this number gets compared
// against monolith_main.cpp's equivalent measurement.

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "deinterleaver.h"
#include "framing.h"
#include "pulse.pb.h"

int main(int argc, char** argv) {
  const uint16_t listen_port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50055;
  constexpr double kSampleRateHz = 10000000.0;
  constexpr double kPriToleranceSeconds = 1e-7;

  const int listen_fd = netutil::Listen(listen_port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[deinterleave_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf("[deinterleave_service] listening on 127.0.0.1:%u, waiting for stats_service...\n",
              listen_port);

  const int upstream_fd = netutil::Accept(listen_fd);
  if (upstream_fd < 0) {
    std::fprintf(stderr, "[deinterleave_service] accept failed\n");
    return 1;
  }
  std::printf("[deinterleave_service] stats_service connected\n");

  pulsecore::Deinterleaver deinterleaver(kSampleRateHz, kPriToleranceSeconds);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PipelineFrame frame;
  int frames_received = 0;
  // See spectrogram_service/main.cpp for why the timer starts on the
  // first successful receive rather than before the loop -- here that
  // choice is what makes this the pipeline-wide steady-state number (see
  // this file's header comment).
  std::chrono::steady_clock::time_point steady_state_start;
  std::chrono::steady_clock::time_point steady_state_end;
  bool started = false;
  while (netutil::RecvMessage(upstream_fd, &payload)) {
    if (!started) {
      steady_state_start = std::chrono::steady_clock::now();
      started = true;
    }
    // In-place clear + merge-parse instead of ParseFromString() -- see
    // spectrogram_service/main.cpp for why (reuses the parsed
    // PulseEvent objects across batches instead of re-allocating them).
    if (frame.has_iq()) frame.mutable_iq()->Clear();
    if (frame.has_events()) frame.mutable_events()->Clear();
    if (!frame.MergeFromString(payload)) {
      std::fprintf(stderr, "[deinterleave_service] dropping malformed frame\n");
      continue;
    }
    deinterleaver.Process(frame.events(), frame.mutable_deinterleave());
    ++frames_received;
  }
  steady_state_end = std::chrono::steady_clock::now();
  const double steady_state_ms =
      started
          ? std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count()
          : 0.0;

  std::printf("[deinterleave_service] received %d frame(s) over TCP, end of chain\n",
              frames_received);
  std::printf("[deinterleave_service] STEADY_STATE_MS %.6f\n", steady_state_ms);
  std::printf("[deinterleave_service] %d track(s)\n", frame.deinterleave().tracks_size());
  for (const pulse::EmitterTrack& track : frame.deinterleave().tracks()) {
    std::printf(
        "[deinterleave_service]   track %u: pulses=%llu estimated_pri_us=%.2f mean_peak=%.3f\n",
        track.track_id(), static_cast<unsigned long long>(track.pulse_count()),
        track.estimated_pri_seconds() * 1e6, track.mean_peak_amplitude());
  }

  ::close(upstream_fd);
  ::close(listen_fd);
  return 0;
}
