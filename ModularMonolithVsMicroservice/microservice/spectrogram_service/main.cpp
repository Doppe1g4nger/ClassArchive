// spectrogram_service: standalone executable running the exact same
// pulsecore::SpectrogramAnalyzer used by libpulse_spectrogram_plugin.so in
// the monolith build. It listens on a TCP port, accepts one connection
// from detector_service, and folds every IQBatch it receives into a
// running magnitude spectrum until the peer closes the connection.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "pulse.pb.h"
#include "spectrogram.h"

int main(int argc, char** argv) {
  const uint16_t port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : 50052;
  constexpr double kSampleRateHz = 1000000.0;
  constexpr int kNumBins = 8;

  const int listen_fd = netutil::Listen(port);
  if (listen_fd < 0) {
    std::fprintf(stderr, "[spectrogram_service] failed to listen on port %u\n", port);
    return 1;
  }
  std::printf("[spectrogram_service] listening on 127.0.0.1:%u, waiting for detector_service...\n",
              port);

  const int client_fd = netutil::Accept(listen_fd);
  if (client_fd < 0) {
    std::fprintf(stderr, "[spectrogram_service] accept failed\n");
    return 1;
  }
  std::printf("[spectrogram_service] detector_service connected\n");

  pulsecore::SpectrogramAnalyzer analyzer(kSampleRateHz, kNumBins);
  std::string payload;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::IQBatch batch;
  pulse::SpectrogramSummary summary;
  int batches_received = 0;
  while (netutil::RecvMessage(client_fd, &payload)) {
    if (!batch.ParseFromString(payload)) {
      std::fprintf(stderr, "[spectrogram_service] dropping malformed message\n");
      continue;
    }
    analyzer.Process(batch, &summary);
    ++batches_received;
  }

  std::printf("[spectrogram_service] received %d IQBatch message(s) over TCP\n",
              batches_received);
  std::printf("[spectrogram_service] %d bins, %.1f Hz spacing, %llu frames\n",
              summary.max_magnitude_size(), summary.bin_hz(),
              static_cast<unsigned long long>(summary.frame_count()));
  for (int i = 0; i < summary.max_magnitude_size(); ++i) {
    std::printf("[spectrogram_service]   bin %d (~%.0f Hz): max=%.3f mean=%.3f\n", i,
                (i + 0.5) * summary.bin_hz(), summary.max_magnitude(i), summary.mean_magnitude(i));
  }

  ::close(client_fd);
  ::close(listen_fd);
  return 0;
}
