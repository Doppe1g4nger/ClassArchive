// stats_service: standalone executable running the exact same
// pulsecore::PulseStatsAccumulator used by libpulse_stats_plugin.so in the
// monolith build. It listens on a TCP port, accepts one connection from
// detector_service, and accumulates every PulseEventBatch it receives
// until the peer closes the connection.

#include <unistd.h>

#include <cstdio>
#include <string>

#include "framing.h"
#include "pulse.pb.h"
#include "pulse_stats.h"

int main(int argc, char** argv) {
  const uint16_t port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50051;
  constexpr double kSampleRateHz = 1000000.0;

  const int listen_fd = netutil::Listen(port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[stats_service] failed to listen on port %u\n", port);
    return 1;
  }
  std::printf("[stats_service] listening on 127.0.0.1:%u, waiting for detector_service...\n",
              port);

  const int client_fd = netutil::Accept(listen_fd);
  if (client_fd < 0) {
    std::fprintf(stderr, "[stats_service] accept failed\n");
    return 1;
  }
  std::printf("[stats_service] detector_service connected\n");

  pulsecore::PulseStatsAccumulator accumulator(kSampleRateHz);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PulseEventBatch events;
  int batches_received = 0;
  while (netutil::RecvMessage(client_fd, &payload)) {
    if (!events.ParseFromString(payload)) {
      std::fprintf(stderr, "[stats_service] dropping malformed message\n");
      continue;
    }
    accumulator.Add(events);
    ++batches_received;
  }

  const pulse::PulseSummary summary = accumulator.Finalize();
  std::printf("[stats_service] received %d PulseEventBatch message(s) over TCP\n",
              batches_received);
  std::printf(
      "[stats_service] pulses=%llu mean_peak=%.3f mean_dur_us=%.2f mean_pri_us=%.2f "
      "min_peak=%.3f max_peak=%.3f\n",
      static_cast<unsigned long long>(summary.pulse_count()), summary.mean_peak_amplitude(),
      summary.mean_duration_seconds() * 1e6, summary.mean_pri_seconds() * 1e6,
      summary.min_peak_amplitude(), summary.max_peak_amplitude());

  ::close(client_fd);
  ::close(listen_fd);
  return 0;
}
