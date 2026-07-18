// deinterleave_service: standalone executable running the exact same
// pulsecore::Deinterleaver used by libpulse_deinterleaver_plugin.so in the
// monolith build. It listens on a TCP port, accepts one connection from
// detector_service, and folds every PulseEventBatch it receives into
// candidate emitter tracks until the peer closes the connection.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "deinterleaver.h"
#include "framing.h"
#include "pulse.pb.h"

int main(int argc, char** argv) {
  const uint16_t port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50054;
  constexpr double kSampleRateHz = 1000000.0;
  constexpr double kPriToleranceSeconds = 5e-6;

  const int listen_fd = netutil::Listen(port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[deinterleave_service] failed to listen on port %u\n", port);
    return 1;
  }
  std::printf("[deinterleave_service] listening on 127.0.0.1:%u, waiting for detector_service...\n",
              port);

  const int client_fd = netutil::Accept(listen_fd);
  if (client_fd < 0) {
    std::fprintf(stderr, "[deinterleave_service] accept failed\n");
    return 1;
  }
  std::printf("[deinterleave_service] detector_service connected\n");

  pulsecore::Deinterleaver deinterleaver(kSampleRateHz, kPriToleranceSeconds);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PulseEventBatch batch;
  pulse::DeinterleaveSummary summary;
  int batches_received = 0;
  while (netutil::RecvMessage(client_fd, &payload)) {
    if (!batch.ParseFromString(payload)) {
      std::fprintf(stderr, "[deinterleave_service] dropping malformed message\n");
      continue;
    }
    deinterleaver.Process(batch, &summary);
    ++batches_received;
  }

  std::printf("[deinterleave_service] received %d PulseEventBatch message(s) over TCP\n",
              batches_received);
  std::printf("[deinterleave_service] %d track(s)\n", summary.tracks_size());
  for (const pulse::EmitterTrack& track : summary.tracks()) {
    std::printf(
        "[deinterleave_service]   track %u: pulses=%llu estimated_pri_us=%.2f mean_peak=%.3f\n",
        track.track_id(), static_cast<unsigned long long>(track.pulse_count()),
        track.estimated_pri_seconds() * 1e6, track.mean_peak_amplitude());
  }

  ::close(client_fd);
  ::close(listen_fd);
  return 0;
}
