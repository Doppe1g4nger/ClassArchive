// spectrogram_service: standalone executable running the exact same
// pulsecore::SpectrogramAnalyzer used by libpulse_spectrogram_plugin.so in
// the monolith build. Second stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver): connects out to
// jammer_service (the next stage) at startup, then listens for
// detector_service. For each frame it receives: folds frame.iq() into a
// running magnitude spectrum, and forwards the frame downstream with its
// own spectrogram field cleared before sending -- jammer_service doesn't
// read it, so there's no reason to pay to serialize and transmit it.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "pulse.pb.h"
#include "spectrogram.h"

int main(int argc, char** argv) {
  const uint16_t listen_port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50052;
  const std::string next_host = argc > 2 ? argv[2] : "127.0.0.1";
  const uint16_t next_port = argc > 3 ? static_cast<uint16_t>(std::atoi(argv[3])) : 50053;
  constexpr double kSampleRateHz = 10000000.0;
  constexpr int kNumBins = 8;

  std::printf("[spectrogram_service] connecting to jammer_service at %s:%u\n", next_host.c_str(),
              next_port);
  const int downstream_fd = netutil::Connect(next_host, next_port);
  if (downstream_fd < 0) {
    std::fprintf(stderr, "[spectrogram_service] failed to connect (is jammer_service running?)\n");
    return 1;
  }

  const int listen_fd = netutil::Listen(listen_port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[spectrogram_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf("[spectrogram_service] listening on 127.0.0.1:%u, waiting for detector_service...\n",
              listen_port);

  const int upstream_fd = netutil::Accept(listen_fd);
  if (upstream_fd < 0) {
    std::fprintf(stderr, "[spectrogram_service] accept failed\n");
    return 1;
  }
  std::printf("[spectrogram_service] detector_service connected\n");

  pulsecore::SpectrogramAnalyzer analyzer(kSampleRateHz, kNumBins);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PipelineFrame frame;
  // Kept separately from frame because frame.spectrogram() gets cleared
  // before every forward (see below) -- this is what "received/forwarded
  // N frames" reports after the loop ends.
  pulse::SpectrogramSummary last_summary;
  int frames_forwarded = 0;
  while (netutil::RecvMessage(upstream_fd, &payload)) {
    if (!frame.ParseFromString(payload)) {
      std::fprintf(stderr, "[spectrogram_service] dropping malformed frame\n");
      continue;
    }
    analyzer.Process(frame.iq(), frame.mutable_spectrogram());
    last_summary = frame.spectrogram();

    // jammer_service (next hop) only reads frame.iq(); nothing downstream
    // of it ever reads frame.spectrogram(), so there's no reason to keep
    // paying to serialize and transmit it past this point.
    frame.clear_spectrogram();

    frame.SerializeToString(&payload);
    if (!netutil::SendMessage(downstream_fd, payload)) {
      std::fprintf(stderr, "[spectrogram_service] forward failed, jammer_service may have exited\n");
      break;
    }
    ++frames_forwarded;
  }

  std::printf("[spectrogram_service] received/forwarded %d frame(s) over TCP\n", frames_forwarded);
  std::printf("[spectrogram_service] %d bins, %.1f Hz spacing, %llu frames\n",
              last_summary.max_magnitude_size(), last_summary.bin_hz(),
              static_cast<unsigned long long>(last_summary.frame_count()));
  for (int i = 0; i < last_summary.max_magnitude_size(); ++i) {
    std::printf("[spectrogram_service]   bin %d (~%.0f Hz): max=%.3f mean=%.3f\n", i,
                (i + 0.5) * last_summary.bin_hz(), last_summary.max_magnitude(i),
                last_summary.mean_magnitude(i));
  }

  ::close(upstream_fd);
  ::close(downstream_fd);
  ::close(listen_fd);
  return 0;
}
