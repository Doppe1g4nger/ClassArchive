// detector_service: standalone executable running the exact same
// pulsecore::PulseDetector used by libpulse_detector_plugin.so in the
// monolith build. It's the data-plane hub of the microservice build: it
// generates the synthetic IQ stream, runs detection locally, and fans
// both the raw batch and the derived pulse events out to four downstream
// services over four separate TCP connections -- one send per consumer,
// since a real process boundary has no broadcast primitive of its own.
//
// Ports are base_port + a fixed offset per consumer (see kPortOffset*
// below) so the CLI only needs one port number. scripts/run_microservices.sh
// starts every listener at those offsets before launching this executable.

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "framing.h"
#include "iq_source.h"
#include "pulse.pb.h"
#include "pulse_detector.h"

namespace {

constexpr int kPortOffsetStats = 0;
constexpr int kPortOffsetSpectrogram = 1;
constexpr int kPortOffsetJammer = 2;
constexpr int kPortOffsetDeinterleave = 3;

int ConnectOrDie(const std::string& host, uint16_t port, const char* consumer_name) {
  std::printf("[detector_service] connecting to %s at %s:%u\n", consumer_name, host.c_str(), port);
  const int fd = netutil::Connect(host, port);
  if (fd < 0) {
    std::fprintf(stderr, "[detector_service] failed to connect to %s (is it running?)\n",
                 consumer_name);
    std::exit(1);
  }
  return fd;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string host = argc > 1 ? argv[1] : "127.0.0.1";
  const uint16_t base_port = argc > 2 ? static_cast<uint16_t>(std::atoi(argv[2])) : 50051;
  const int num_pulses = argc > 3 ? std::atoi(argv[3]) : 6;

  const int stats_fd = ConnectOrDie(host, base_port + kPortOffsetStats, "stats_service");
  const int spectrogram_fd =
      ConnectOrDie(host, base_port + kPortOffsetSpectrogram, "spectrogram_service");
  const int jammer_fd = ConnectOrDie(host, base_port + kPortOffsetJammer, "jammer_service");
  const int deinterleave_fd =
      ConnectOrDie(host, base_port + kPortOffsetDeinterleave, "deinterleave_service");

  constexpr double kSampleRateHz = 1000000.0;
  pulsecore::PulseDetector detector(/*amplitude_threshold=*/6.0, kSampleRateHz);
  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);

  pulse::IQBatch iq_batch;
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp.
  pulse::PulseEventBatch events;
  std::string iq_payload;
  std::string events_payload;
  int batches_sent = 0;

  while (source.NextBatch(&iq_batch)) {
    events.Clear();
    detector.Process(iq_batch, &events);

    // spectrogram_service and jammer_service both analyze the raw batch;
    // stats_service and deinterleave_service both analyze the pulses the
    // detector found in it. Each connection gets its own serialize+send
    // -- there's no way around paying for it twice per message type
    // since these are four independent processes, not four listeners on
    // one broadcast.
    iq_batch.SerializeToString(&iq_payload);
    events.SerializeToString(&events_payload);

    const bool ok = netutil::SendMessage(stats_fd, events_payload) &&
                     netutil::SendMessage(deinterleave_fd, events_payload) &&
                     netutil::SendMessage(spectrogram_fd, iq_payload) &&
                     netutil::SendMessage(jammer_fd, iq_payload);
    if (!ok) {
      std::fprintf(stderr, "[detector_service] send failed, a consumer may have exited\n");
      ::close(stats_fd);
      ::close(spectrogram_fd);
      ::close(jammer_fd);
      ::close(deinterleave_fd);
      return 1;
    }
    ++batches_sent;
  }

  std::printf("[detector_service] streamed %d batch(es) to all four consumers, closing\n",
              batches_sent);
  ::close(stats_fd);
  ::close(spectrogram_fd);
  ::close(jammer_fd);
  ::close(deinterleave_fd);
  return 0;
}
