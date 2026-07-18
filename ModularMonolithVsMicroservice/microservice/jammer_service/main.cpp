// jammer_service: standalone executable running the exact same
// pulsecore::JammerDetector used by libpulse_jammer_plugin.so in the
// monolith build. Fifth and final stage of the pipeline chain (detector
// -> stats -> deinterleaver -> spectrogram -> jammer): it's the chain's
// sink, so it only listens for spectrogram_service and never forwards --
// it folds frame.iq() into running jam-detection state and prints the
// result once the upstream connection closes.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "jammer.h"
#include "pulse.pb.h"

int main(int argc, char** argv) {
  const uint16_t listen_port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50054;
  constexpr double kPowerThreshold = 20.0;
  constexpr double kDutyCycleThreshold = 0.5;

  const int listen_fd = netutil::Listen(listen_port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[jammer_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf("[jammer_service] listening on 127.0.0.1:%u, waiting for spectrogram_service...\n",
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
  int frames_received = 0;
  while (netutil::RecvMessage(upstream_fd, &payload)) {
    if (!frame.ParseFromString(payload)) {
      std::fprintf(stderr, "[jammer_service] dropping malformed frame\n");
      continue;
    }
    detector.Process(frame.iq(), frame.mutable_jam());
    ++frames_received;
  }

  std::printf("[jammer_service] received %d frame(s) over TCP, end of chain\n", frames_received);
  const pulse::JamSummary& jam = frame.jam();
  std::printf(
      "[jammer_service] %llu/%llu batches flagged, max_duty_cycle=%.3f max_mean_power=%.2f\n",
      static_cast<unsigned long long>(jam.batches_flagged()),
      static_cast<unsigned long long>(jam.batches_total()), jam.max_duty_cycle(),
      jam.max_mean_power());

  ::close(upstream_fd);
  ::close(listen_fd);
  return 0;
}
