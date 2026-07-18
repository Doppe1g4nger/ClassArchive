// detector_service: standalone executable running the exact same
// pulsecore::PulseDetector used by libpulse_detector_plugin.so in the
// monolith build. First stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver): it's the chain's pure
// producer, so it never listens -- it generates the synthetic IQ stream,
// runs detection locally, and connects out to spectrogram_service (the
// next stage) as a plain TCP client, streaming one serialized
// PipelineFrame per batch.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "iq_source.h"
#include "pulse.pb.h"
#include "pulse_detector.h"

int main(int argc, char** argv) {
  const std::string next_host = argc > 1 ? argv[1] : "127.0.0.1";
  const uint16_t next_port = argc > 2 ? static_cast<uint16_t>(std::atoi(argv[2])) : 50051;
  const int num_pulses = argc > 3 ? std::atoi(argv[3]) : 6;

  std::printf("[detector_service] connecting to spectrogram_service at %s:%u\n", next_host.c_str(),
              next_port);
  const int downstream_fd = netutil::Connect(next_host, next_port);
  if (downstream_fd < 0) {
    std::fprintf(stderr,
                  "[detector_service] failed to connect (is spectrogram_service running?)\n");
    return 1;
  }

  constexpr double kSampleRateHz = 1000000.0;
  pulsecore::PulseDetector detector(/*amplitude_threshold=*/6.0, kSampleRateHz);
  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);

  // Reused across iterations for the same reason common/ loops reuse
  // their message objects -- see pulse_detector_plugin.cpp's history.
  // frame.iq() is filled directly by NextBatch() below (no copy).
  pulse::PipelineFrame frame;
  std::string payload;
  int batches_sent = 0;

  while (source.NextBatch(frame.mutable_iq())) {
    frame.mutable_events()->Clear();
    detector.Process(frame.iq(), frame.mutable_events());

    frame.SerializeToString(&payload);
    if (!netutil::SendMessage(downstream_fd, payload)) {
      std::fprintf(stderr, "[detector_service] send failed, stats_service may have exited\n");
      ::close(downstream_fd);
      return 1;
    }
    ++batches_sent;
  }

  std::printf("[detector_service] streamed %d frame(s) into the chain, closing\n", batches_sent);
  ::close(downstream_fd);
  return 0;
}
