// stats_service: standalone executable running the exact same
// pulsecore::PulseStatsAccumulator used by libpulse_stats_plugin.so in the
// monolith build. Second stage of the pipeline chain (detector -> stats
// -> deinterleaver -> spectrogram -> jammer): connects out to
// deinterleave_service (the next stage) at startup, then listens for
// detector_service. For each frame it receives: folds frame.events()
// into running stats, writes the cumulative summary into frame.stats(),
// and forwards the enriched frame downstream.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "pulse.pb.h"
#include "pulse_stats.h"

int main(int argc, char** argv) {
  const uint16_t listen_port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50051;
  const std::string next_host = argc > 2 ? argv[2] : "127.0.0.1";
  const uint16_t next_port = argc > 3 ? static_cast<uint16_t>(std::atoi(argv[3])) : 50052;
  constexpr double kSampleRateHz = 1000000.0;

  std::printf("[stats_service] connecting to deinterleave_service at %s:%u\n", next_host.c_str(),
              next_port);
  const int downstream_fd = netutil::Connect(next_host, next_port);
  if (downstream_fd < 0) {
    std::fprintf(stderr,
                  "[stats_service] failed to connect (is deinterleave_service running?)\n");
    return 1;
  }

  const int listen_fd = netutil::Listen(listen_port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[stats_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf("[stats_service] listening on 127.0.0.1:%u, waiting for detector_service...\n",
              listen_port);

  const int upstream_fd = netutil::Accept(listen_fd);
  if (upstream_fd < 0) {
    std::fprintf(stderr, "[stats_service] accept failed\n");
    return 1;
  }
  std::printf("[stats_service] detector_service connected\n");

  pulsecore::PulseStatsAccumulator accumulator(kSampleRateHz);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PipelineFrame frame;
  int frames_forwarded = 0;
  while (netutil::RecvMessage(upstream_fd, &payload)) {
    if (!frame.ParseFromString(payload)) {
      std::fprintf(stderr, "[stats_service] dropping malformed frame\n");
      continue;
    }
    accumulator.Add(frame.events());
    *frame.mutable_stats() = accumulator.Finalize();

    frame.SerializeToString(&payload);
    if (!netutil::SendMessage(downstream_fd, payload)) {
      std::fprintf(stderr,
                    "[stats_service] forward failed, deinterleave_service may have exited\n");
      break;
    }
    ++frames_forwarded;
  }

  std::printf("[stats_service] received/forwarded %d frame(s) over TCP\n", frames_forwarded);
  const pulse::PulseSummary& summary = frame.stats();
  std::printf(
      "[stats_service] pulses=%llu mean_peak=%.3f mean_dur_us=%.2f mean_pri_us=%.2f "
      "min_peak=%.3f max_peak=%.3f\n",
      static_cast<unsigned long long>(summary.pulse_count()), summary.mean_peak_amplitude(),
      summary.mean_duration_seconds() * 1e6, summary.mean_pri_seconds() * 1e6,
      summary.min_peak_amplitude(), summary.max_peak_amplitude());

  ::close(upstream_fd);
  ::close(downstream_fd);
  ::close(listen_fd);
  return 0;
}
