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

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "deinterleaver.h"
#include "iq_source.h"
#include "jammer_avx.h"
#include "pulse.pb.h"
#include "pulse_detector_avx.h"
#include "pulse_stats.h"
#include "spectrogram.h"
#include "threaded_pipeline.h"

int main(int argc, char** argv) {
  // Default of 1000 pulses matches every other build in this repo.
  const int num_pulses = argc > 1 ? std::atoi(argv[1]) : 1000;

  constexpr double kSampleRateHz = 10000000.0;
  constexpr int kNumBins = 8;

  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);
  pulsecore::PulseDetectorAvx detector(/*amplitude_threshold=*/6.0, kSampleRateHz);
  // Two half-range spectrogram instances, one per thread -- same split
  // (and same reasons) as monolith_main.cpp.
  pulsecore::SpectrogramAnalyzer spectrogram_lo(kSampleRateHz, kNumBins, 0, kNumBins / 2);
  pulsecore::SpectrogramAnalyzer spectrogram_hi(kSampleRateHz, kNumBins, kNumBins / 2, kNumBins);
  pulsecore::JammerDetectorAvx jammer(/*power_threshold=*/20.0, /*duty_cycle_threshold=*/0.5);
  pulsecore::PulseStatsAccumulator stats_accumulator(kSampleRateHz);
  pulsecore::Deinterleaver deinterleaver(kSampleRateHz, /*pri_tolerance_seconds=*/1e-7);

  // Round three: the signal is a given -- all batches pre-generated
  // before the clock starts, measured pipeline begins at detection.
  // Same stage-per-thread harness and grouping as monolith_main.cpp
  // (see both for why), just direct calls instead of module function
  // pointers.
  std::vector<pulse::PipelineFrame> frames;
  {
    pulse::PipelineFrame f;
    while (source.NextBatch(f.mutable_iq())) {
      frames.push_back(std::move(f));
      f.Clear();
    }
  }
  // Pre-size outputs untimed -- see monolith_main.cpp for why (the
  // spectrogram arrays especially: the two half-range instances write
  // disjoint entries concurrently and must never resize).
  for (pulse::PipelineFrame& f : frames) {
    pulse::PulseEventBatch* ev = f.mutable_events();
    ev->mutable_start_sample()->Reserve(2048);
    ev->mutable_end_sample()->Reserve(2048);
    ev->mutable_peak_amplitude()->Reserve(2048);
    ev->mutable_mean_amplitude()->Reserve(2048);
    ev->mutable_duration_seconds()->Reserve(2048);
    pulse::SpectrogramSummary* spec = f.mutable_spectrogram();
    for (int b = 0; b < kNumBins; ++b) {
      spec->add_max_magnitude(0.0);
      spec->add_mean_magnitude(0.0);
    }
    f.mutable_jam();
    f.mutable_stats();
    f.mutable_deinterleave();
  }

  std::vector<std::function<void(pulse::PipelineFrame*)>> stages = {
      [&](pulse::PipelineFrame* f) {
        f->mutable_events()->Clear();
        detector.Process(f->iq(), f->mutable_events());
      },
      [&](pulse::PipelineFrame* f) { spectrogram_lo.Process(f->iq(), f->mutable_spectrogram()); },
      [&](pulse::PipelineFrame* f) { spectrogram_hi.Process(f->iq(), f->mutable_spectrogram()); },
      [&](pulse::PipelineFrame* f) {
        jammer.Process(f->iq(), f->mutable_jam());
        stats_accumulator.Add(f->events());
        *f->mutable_stats() = stats_accumulator.Finalize();
        deinterleaver.Process(f->events(), f->mutable_deinterleave());
      },
  };

  // Timed region covers detection through deinterleave, same charter
  // as every other build in this repo.
  const monolith::ThreadedPipelineResult run = monolith::RunThreadedPipeline(stages, &frames);
  static const pulse::PipelineFrame kEmptyFrame;
  const pulse::PipelineFrame& frame = run.final_frame ? *run.final_frame : kEmptyFrame;

  std::printf("[avx_monolith_app] processed %d IQ batches through 2 AVX2 kernels + 3 scalar stages\n",
              run.batches);
  std::printf("[avx_monolith_app] STEADY_STATE_MS %.6f\n", run.steady_state_ms);

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
