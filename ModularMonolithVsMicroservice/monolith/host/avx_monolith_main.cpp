// avx_monolith_app: a fourth C++ architecture, answering "optimize
// further with AVX instructions?" from the correctness-audit follow-up.
// Same five-stage pipeline as monolith_main.cpp -- detector -> spectrogram
// -> jammer -> stats -> deinterleaver -- but statically linked instead of
// dlopen()'d (there's nothing "modular" being demonstrated here, so
// there's no reason to pay the dlsym() indirection) and with the
// detector and jammer stages replaced by their AVX2-vectorized ports
// (pulse_detector_avx.h/jammer_avx.h). Spectrogram, stats, and
// deinterleaver are unmodified pulsecore code -- see those headers for
// why AVX wasn't applied to the spectrogram (no portable
// dependency-free vectorized cos()/sin()) and why the detector's output
// is bit-identical to monolith_app's while the jammer's isn't (see
// scripts/verify_avx_variant.sh for the measured numbers, not just the
// claim).

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "deinterleaver.h"
#include "iq_source.h"
#include "jammer_avx.h"
#include "pulse.pb.h"
#include "pulse_detector_avx.h"
#include "pulse_stats.h"
#include "spectrogram.h"

int main(int argc, char** argv) {
  // Default of 1000 pulses matches every other build in this repo.
  const int num_pulses = argc > 1 ? std::atoi(argv[1]) : 1000;

  constexpr double kSampleRateHz = 10000000.0;
  constexpr int kNumBins = 8;

  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);
  pulsecore::PulseDetectorAvx detector(/*amplitude_threshold=*/6.0, kSampleRateHz);
  pulsecore::SpectrogramAnalyzer spectrogram(kSampleRateHz, kNumBins);
  pulsecore::JammerDetectorAvx jammer(/*power_threshold=*/20.0, /*duty_cycle_threshold=*/0.5);
  pulsecore::PulseStatsAccumulator stats_accumulator(kSampleRateHz);
  pulsecore::Deinterleaver deinterleaver(kSampleRateHz, /*pri_tolerance_seconds=*/1e-7);

  // Reused across iterations for the same reason monolith_main.cpp's
  // frame is.
  pulse::PipelineFrame frame;
  int batches = 0;

  // Timed region covers only the batch-processing loop, same convention
  // as every other build in this repo.
  const auto steady_state_start = std::chrono::steady_clock::now();
  while (source.NextBatch(frame.mutable_iq())) {
    frame.mutable_events()->Clear();
    detector.Process(frame.iq(), frame.mutable_events());
    spectrogram.Process(frame.iq(), frame.mutable_spectrogram());
    jammer.Process(frame.iq(), frame.mutable_jam());
    stats_accumulator.Add(frame.events());
    *frame.mutable_stats() = stats_accumulator.Finalize();
    deinterleaver.Process(frame.events(), frame.mutable_deinterleave());
    ++batches;
  }
  const auto steady_state_end = std::chrono::steady_clock::now();
  const double steady_state_ms =
      std::chrono::duration<double, std::milli>(steady_state_end - steady_state_start).count();

  std::printf("[avx_monolith_app] processed %d IQ batches through 2 AVX2 kernels + 3 scalar stages\n",
              batches);
  std::printf("[avx_monolith_app] STEADY_STATE_MS %.6f\n", steady_state_ms);

  const pulse::SpectrogramSummary& spec = frame.spectrogram();
  std::printf("[avx_monolith_app] spectrogram: %d bins, %.1f Hz spacing, %llu frames\n",
              spec.max_magnitude_size(), spec.bin_hz(),
              static_cast<unsigned long long>(spec.frame_count()));
  for (int i = 0; i < spec.max_magnitude_size(); ++i) {
    std::printf("[avx_monolith_app]   bin %d (~%.0f Hz): max=%.3f mean=%.3f\n", i,
                (i + 0.5) * spec.bin_hz(), spec.max_magnitude(i), spec.mean_magnitude(i));
  }

  const pulse::JamSummary& jam = frame.jam();
  std::printf(
      "[avx_monolith_app] jammer: %llu/%llu batches flagged, max_duty_cycle=%.3f max_mean_power=%.2f\n",
      static_cast<unsigned long long>(jam.batches_flagged()),
      static_cast<unsigned long long>(jam.batches_total()), jam.max_duty_cycle(),
      jam.max_mean_power());

  const pulse::PulseSummary& stats = frame.stats();
  std::printf(
      "[avx_monolith_app] stats: pulses=%llu mean_peak=%.3f mean_dur_us=%.2f mean_pri_us=%.2f "
      "min_peak=%.3f max_peak=%.3f\n",
      static_cast<unsigned long long>(stats.pulse_count()), stats.mean_peak_amplitude(),
      stats.mean_duration_seconds() * 1e6, stats.mean_pri_seconds() * 1e6,
      stats.min_peak_amplitude(), stats.max_peak_amplitude());

  const pulse::DeinterleaveSummary& tracks = frame.deinterleave();
  std::printf("[avx_monolith_app] deinterleaver: %d track(s)\n", tracks.tracks_size());
  for (const pulse::EmitterTrack& track : tracks.tracks()) {
    std::printf("[avx_monolith_app]   track %u: pulses=%llu estimated_pri_us=%.2f mean_peak=%.3f\n",
                track.track_id(), static_cast<unsigned long long>(track.pulse_count()),
                track.estimated_pri_seconds() * 1e6, track.mean_peak_amplitude());
  }

  return 0;
}
