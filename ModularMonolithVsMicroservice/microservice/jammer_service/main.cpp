// jammer_service: standalone executable running the exact same
// pulsecore::JammerDetector used by libpulse_jammer_plugin.so in the
// monolith build. It listens on a TCP port, accepts one connection from
// detector_service, and folds every IQBatch it receives into running
// jam-detection state until the peer closes the connection.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "jammer.h"
#include "pulse.pb.h"

int main(int argc, char** argv) {
  const uint16_t port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50053;
  constexpr double kPowerThreshold = 20.0;
  constexpr double kDutyCycleThreshold = 0.5;

  const int listen_fd = netutil::Listen(port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[jammer_service] failed to listen on port %u\n", port);
    return 1;
  }
  std::printf("[jammer_service] listening on 127.0.0.1:%u, waiting for detector_service...\n",
              port);

  const int client_fd = netutil::Accept(listen_fd);
  if (client_fd < 0) {
    std::fprintf(stderr, "[jammer_service] accept failed\n");
    return 1;
  }
  std::printf("[jammer_service] detector_service connected\n");

  pulsecore::JammerDetector detector(kPowerThreshold, kDutyCycleThreshold);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::IQBatch batch;
  pulse::JamSummary summary;
  int batches_received = 0;
  while (netutil::RecvMessage(client_fd, &payload)) {
    if (!batch.ParseFromString(payload)) {
      std::fprintf(stderr, "[jammer_service] dropping malformed message\n");
      continue;
    }
    detector.Process(batch, &summary);
    ++batches_received;
  }

  std::printf("[jammer_service] received %d IQBatch message(s) over TCP\n", batches_received);
  std::printf(
      "[jammer_service] %llu/%llu batches flagged, max_duty_cycle=%.3f max_mean_power=%.2f\n",
      static_cast<unsigned long long>(summary.batches_flagged()),
      static_cast<unsigned long long>(summary.batches_total()), summary.max_duty_cycle(),
      summary.max_mean_power());

  ::close(client_fd);
  ::close(listen_fd);
  return 0;
}
