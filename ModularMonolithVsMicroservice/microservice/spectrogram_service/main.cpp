// spectrogram_service: standalone executable running the exact same
// pulsecore::SpectrogramAnalyzer used by libpulse_spectrogram_plugin.so in
// the monolith build. Second stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver): connects out to
// jammer_service (the next stage) at startup, then listens for
// detector_service. For each frame it receives: folds frame.iq() into a
// running magnitude spectrum, and forwards the frame downstream with its
// own spectrogram field cleared before sending -- jammer_service doesn't
// read it, so there's no reason to pay to serialize and transmit it.

#include <chrono>
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
  netutil::Channel* downstream = netutil::Connect(next_host, next_port);
  if (downstream == nullptr) {
    std::fprintf(stderr, "[spectrogram_service] failed to connect (is jammer_service running?)\n");
    return 1;
  }

  netutil::Channel* listener = netutil::Listen(listen_port);
  if (listener == nullptr) {
    std::fprintf(stderr, "[spectrogram_service] failed to listen on port %u\n", listen_port);
    return 1;
  }
  std::printf("[spectrogram_service] listening on shm ring %u, waiting for detector_service...\n",
              listen_port);

  netutil::Channel* upstream = netutil::Accept(listener);
  if (upstream == nullptr) {
    std::fprintf(stderr, "[spectrogram_service] accept failed\n");
    return 1;
  }
  std::printf("[spectrogram_service] detector_service connected\n");

  pulsecore::SpectrogramAnalyzer analyzer(kSampleRateHz, kNumBins);
  // Reused across iterations for the same reason the plugin modules reuse
  // theirs -- see pulse_detector_plugin.cpp. No payload string anymore:
  // round three's zero-copy framing serializes/parses directly in the
  // ring slots (see framing.h).
  pulse::PipelineFrame frame;
  // Kept separately from frame because frame.spectrogram() gets cleared
  // before every forward (see below) -- this is what "received/forwarded
  // N frames" reports after the loop ends.
  pulse::SpectrogramSummary last_summary;
  int frames_forwarded = 0;
  // Set on the first successful receive, not before -- the wait for that
  // first message is this stage's share of the chain's connection-setup
  // latency (detector_service can only finish connecting once
  // spectrogram_service is listening, but nothing actually flows until
  // every downstream hop is ready too), which we want excluded from a
  // steady-state measurement the same way monolith_main.cpp excludes
  // dlopen(). See detector_service/main.cpp and
  // deinterleave_service/main.cpp for the same pattern at the other ends
  // of the pipeline.
  std::chrono::steady_clock::time_point steady_state_start;
  std::chrono::steady_clock::time_point steady_state_end;
  bool started = false;
  for (;;) {
    // Clear the two big repeated-field carriers *in place* BEFORE the
    // merge-parse (which now happens inside RecvMessage, straight from
    // the ring slot). The reasoning is the max-opt branch's: a
    // non-merge parse would run the generated Clear() first, and for
    // singular message fields that Clear() *deletes* the submessage
    // outright, destroying and re-heap-allocating every parsed object
    // each batch (callgrind measured that churn at ~45% of this
    // process's instructions before the fix). In-place Clear() zeroes
    // and CACHES; the merge lands in the cached objects. The upstream
    // stages only ever send iq/events (each stage strips its own
    // summary before forwarding), so these two clears cover everything
    // the wire can carry here.
    if (frame.has_iq()) frame.mutable_iq()->Clear();
    if (frame.has_events()) frame.mutable_events()->Clear();
    if (!netutil::RecvMessage(upstream, &frame)) break;
    if (!started) {
      steady_state_start = std::chrono::steady_clock::now();
      started = true;
    }
    analyzer.Process(frame.iq(), frame.mutable_spectrogram());
    last_summary = frame.spectrogram();

    // jammer_service (next hop) only reads frame.iq(); nothing downstream
    // of it ever reads frame.spectrogram(), so there's no reason to keep
    // paying to serialize and transmit it past this point. In-place
    // Clear() rather than clear_spectrogram(), which would *delete* the
    // submessage (and its two heap-allocated repeated-double fields)
    // every batch -- see jammer_service/main.cpp for the full story on
    // this codegen behavior; the wire cost of the resulting
    // present-but-empty field is 2 bytes.
    frame.mutable_spectrogram()->Clear();

    // Zero-copy forward: serializes directly into the downstream slot.
    if (!netutil::SendMessage(downstream, frame)) {
      std::fprintf(stderr, "[spectrogram_service] forward failed, jammer_service may have exited\n");
      break;
    }
    ++frames_forwarded;
  }
  steady_state_end = std::chrono::steady_clock::now();
  const double steady_state_ms =
      started
          ? std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count()
          : 0.0;

  std::printf("[spectrogram_service] received/forwarded %d frame(s) via shm ring\n", frames_forwarded);
  std::printf("[spectrogram_service] STEADY_STATE_MS %.6f\n", steady_state_ms);
  std::printf("[spectrogram_service] %d bins, %.1f Hz spacing, %llu frames\n",
              last_summary.max_magnitude_size(), last_summary.bin_hz(),
              static_cast<unsigned long long>(last_summary.frame_count()));
  for (int i = 0; i < last_summary.max_magnitude_size(); ++i) {
    std::printf("[spectrogram_service]   bin %d (~%.0f Hz): max=%.3f mean=%.3f\n", i,
                (i + 0.5) * last_summary.bin_hz(), last_summary.max_magnitude(i),
                last_summary.mean_magnitude(i));
  }

  netutil::Close(upstream);  // == listener; consumer side unlinks the ring
  netutil::Close(downstream);
  return 0;
}
