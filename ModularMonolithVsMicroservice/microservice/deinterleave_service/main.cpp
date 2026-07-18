// deinterleave_service: standalone executable running the exact same
// pulsecore::Deinterleaver used by libpulse_deinterleaver_plugin.so in the
// monolith build. Third stage of the pipeline chain (detector -> stats ->
// deinterleaver -> spectrogram -> jammer): connects out to
// spectrogram_service (the next stage) at startup, then listens for
// stats_service. For each frame it receives: folds frame.events() into
// candidate emitter tracks, writes them into frame.deinterleave(), and
// forwards the enriched frame downstream.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "deinterleaver.h"
#include "framing.h"
#include "pulse.pb.h"

int main(int argc, char** argv) {
  const uint16_t listen_port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50052;
  const std::string next_host = argc > 2 ? argv[2] : "127.0.0.1";
  const uint16_t next_port = argc > 3 ? static_cast<uint16_t>(std::atoi(argv[3])) : 50053;
  constexpr double kSampleRateHz = 1000000.0;
  constexpr double kPriToleranceSeconds = 5e-6;

  std::printf("[deinterleave_service] connecting to spectrogram_service at %s:%u\n",
              next_host.c_str(), next_port);
  const int downstream_fd = netutil::Connect(next_host, next_port);
  if (downstream_fd < 0) {
    std::fprintf(stderr,
                  "[deinterleave_service] failed to connect (is spectrogram_service running?)\n");
    return 1;
  }

  const int listen_fd = netutil::Listen(listen_port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[deinterleave_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf(
      "[deinterleave_service] listening on 127.0.0.1:%u, waiting for stats_service...\n",
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
  int frames_forwarded = 0;
  while (netutil::RecvMessage(upstream_fd, &payload)) {
    if (!frame.ParseFromString(payload)) {
      std::fprintf(stderr, "[deinterleave_service] dropping malformed frame\n");
      continue;
    }
    deinterleaver.Process(frame.events(), frame.mutable_deinterleave());

    frame.SerializeToString(&payload);
    if (!netutil::SendMessage(downstream_fd, payload)) {
      std::fprintf(stderr,
                    "[deinterleave_service] forward failed, spectrogram_service may have exited\n");
      break;
    }
    ++frames_forwarded;
  }

  std::printf("[deinterleave_service] received/forwarded %d frame(s) over TCP\n",
              frames_forwarded);
  std::printf("[deinterleave_service] %d track(s)\n", frame.deinterleave().tracks_size());
  for (const pulse::EmitterTrack& track : frame.deinterleave().tracks()) {
    std::printf(
        "[deinterleave_service]   track %u: pulses=%llu estimated_pri_us=%.2f mean_peak=%.3f\n",
        track.track_id(), static_cast<unsigned long long>(track.pulse_count()),
        track.estimated_pri_seconds() * 1e6, track.mean_peak_amplitude());
  }

  ::close(upstream_fd);
  ::close(downstream_fd);
  ::close(listen_fd);
  return 0;
}
