// detector_service: standalone executable running the exact same
// pulsecore::PulseDetector used by libpulse_detector_plugin.so in the
// monolith build. Instead of handing its output to another module via a
// dlopen()'d function call, it serializes each PulseEventBatch and sends
// it over a TCP connection to stats_service.

#include <unistd.h>

#include <cstdio>
#include <string>

#include "framing.h"
#include "iq_source.h"
#include "pulse.pb.h"
#include "pulse_detector.h"

int main(int argc, char** argv) {
  const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
  const uint16_t port = argc > 2 ? static_cast<uint16_t>(std::atoi(argv[2])) : 50051;

  std::printf("[detector_service] connecting to stats_service at %s:%u\n", host.c_str(), port);
  const int fd = netutil::Connect(host, port);
  if (fd < 0) {
    std::fprintf(stderr, "[detector_service] failed to connect (is stats_service running?)\n");
    return 1;
  }

  constexpr double kSampleRateHz = 1000000.0;
  pulsecore::PulseDetector detector(/*amplitude_threshold=*/6.0, kSampleRateHz);
  pulsecore::SyntheticIQSource source(kSampleRateHz, /*num_pulses=*/6);

  pulse::IQBatch iq_batch;
  int batches_sent = 0;
  while (source.NextBatch(&iq_batch)) {
    pulse::PulseEventBatch events;
    detector.Process(iq_batch, &events);

    std::string payload;
    events.SerializeToString(&payload);
    if (!netutil::SendMessage(fd, payload)) {
      std::fprintf(stderr, "[detector_service] send failed, stats_service may have exited\n");
      ::close(fd);
      return 1;
    }
    ++batches_sent;
  }

  std::printf("[detector_service] streamed %d PulseEventBatch message(s) over TCP, closing\n",
              batches_sent);
  ::close(fd);
  return 0;
}
