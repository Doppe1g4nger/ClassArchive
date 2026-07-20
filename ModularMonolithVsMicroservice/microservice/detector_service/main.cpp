// detector_service: standalone executable running the exact same
// pulsecore::PulseDetector used by libpulse_detector_plugin.so in the
// monolith build. First stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver): it's the chain's pure
// producer, so it never listens -- it generates the synthetic IQ stream,
// runs detection locally, and connects out to spectrogram_service (the
// next stage) as the producer end of its shm ring (see net/framing.h),
// streaming one serialized PipelineFrame per batch.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "framing.h"
#include "iq_source.h"
#include "pulse.pb.h"
#include "pulse_detector.h"

int main(int argc, char** argv) {
  const std::string next_host = argc > 1 ? argv[1] : "127.0.0.1";
  const uint16_t next_port = argc > 2 ? static_cast<uint16_t>(std::atoi(argv[2])) : 50051;
  // Default of 1000 pulses is exactly one buffer's worth at this repo's
  // 1,000,000-pulse/sec, 1000-microsecond-buffer scale (see iq_source.h).
  const int num_pulses = argc > 3 ? std::atoi(argv[3]) : 1000;

  std::printf("[detector_service] connecting to spectrogram_service at %s:%u\n", next_host.c_str(),
              next_port);
  netutil::Channel* downstream = netutil::Connect(next_host, next_port);
  if (downstream == nullptr) {
    std::fprintf(stderr,
                  "[detector_service] failed to connect (is spectrogram_service running?)\n");
    return 1;
  }

  constexpr double kSampleRateHz = 10000000.0;
  pulsecore::PulseDetector detector(/*amplitude_threshold=*/6.0, kSampleRateHz);
  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);

  // Round three of the theoretical-limits branch: the signal is a
  // GIVEN (real IQ comes from a radio; no architecture makes the
  // antenna faster), so every batch is generated before the clock
  // starts and the measured region begins at detection -- matching
  // monolith_main.cpp's identical charter.
  std::vector<pulse::PipelineFrame> frames;
  {
    pulse::PipelineFrame f;
    while (source.NextBatch(f.mutable_iq())) {
      frames.push_back(std::move(f));
      f.Clear();
    }
  }
  int batches_sent = 0;

  // Timed region starts right after connect() succeeds -- which, thanks
  // to this chain's reverse-order startup (see
  // scripts/run_microservices.sh), can only happen once every downstream
  // hop (spectrogram -> jammer -> stats -> deinterleave) is already
  // listening. So this measurement excludes not just detector_service's
  // own connection setup but the whole chain's, without needing to
  // coordinate timestamps across processes. See
  // microservice/deinterleave_service/main.cpp for the matching
  // measurement at the other end of the pipeline.
  const auto steady_state_start = std::chrono::steady_clock::now();
  for (pulse::PipelineFrame& frame : frames) {
    detector.Process(frame.iq(), frame.mutable_events());

    // Zero-copy send: serializes directly into the ring slot -- no
    // payload string, no zero-fill, no extra copy (see framing.h).
    if (!netutil::SendMessage(downstream, frame)) {
      std::fprintf(stderr, "[detector_service] send failed, spectrogram_service may have exited\n");
      netutil::Close(downstream);
      return 1;
    }
    ++batches_sent;
  }
  const auto steady_state_end = std::chrono::steady_clock::now();
  const double steady_state_ms =
      std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count();

  std::printf("[detector_service] streamed %d frame(s) into the chain, closing\n", batches_sent);
  std::printf("[detector_service] STEADY_STATE_MS %.6f\n", steady_state_ms);
  netutil::Close(downstream);
  return 0;
}
