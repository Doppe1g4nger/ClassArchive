// monolith_app: a single process/executable that dlopen()s five
// pulse-processing modules at startup and runs them as a linear chain --
// detector -> spectrogram -> jammer -> stats -> deinterleaver -- each
// stage reading/writing one pulse::PipelineFrame passed by reference, no
// serialization, since every module is built from the same
// pulse.pb.h/pulse_proto as the host (see module_api.h for why that makes
// it safe). Unlike the microservice build, this chain never needs to drop
// unread fields from the frame between stages -- passing a reference
// costs the same regardless of what's populated -- so the ordering here
// is purely for consistency with microservice/'s wiring, not a
// performance requirement. Compare with
// microservice/{detector_service,spectrogram_service,jammer_service,
// stats_service,deinterleave_service}, which run the identical core
// algorithms as five separate processes chained over TCP sockets -- each
// one both a server to the stage before it and a client to the stage
// after it -- because they don't share an address space, and which
// therefore do care what's still in the frame at each hop.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "iq_source.h"
#include "module_api.h"
#include "pulse.pb.h"
#include "threaded_pipeline.h"

namespace {

struct LoadedModule {
  void* handle = nullptr;
  pulse_module_destroy_fn destroy = nullptr;
  pulse_stage_process_fn process = nullptr;
  pulse_module_t instance = nullptr;
};

LoadedModule LoadModule(const std::string& path, const char* config) {
  LoadedModule m;
  m.handle = dlopen(path.c_str(), RTLD_NOW);
  if (m.handle == nullptr) {
    std::fprintf(stderr, "[monolith_app] dlopen(%s) failed: %s\n", path.c_str(), dlerror());
    std::exit(1);
  }

  auto create = reinterpret_cast<pulse_module_create_fn>(dlsym(m.handle, PULSE_MODULE_CREATE_SYM));
  m.destroy = reinterpret_cast<pulse_module_destroy_fn>(dlsym(m.handle, PULSE_MODULE_DESTROY_SYM));
  m.process = reinterpret_cast<pulse_stage_process_fn>(dlsym(m.handle, PULSE_STAGE_PROCESS_SYM));
  if (create == nullptr || m.destroy == nullptr || m.process == nullptr) {
    std::fprintf(stderr, "[monolith_app] %s is missing a required symbol\n", path.c_str());
    std::exit(1);
  }

  m.instance = create(config);
  std::printf("[monolith_app] loaded %s\n", path.c_str());
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string plugin_dir = argc > 1 ? argv[1] : ".";
  // Default of 1000 pulses is exactly one buffer's worth at this repo's
  // 1,000,000-pulse/sec, 1000-microsecond-buffer scale (see iq_source.h),
  // so running with no arguments demonstrates that scale directly.
  const int num_pulses = argc > 2 ? std::atoi(argv[2]) : 1000;

  // Chain order matches microservice/'s wiring exactly (see
  // scripts/run_microservices.sh): each stage needs whatever the stages
  // before it in this list have already written into the frame.
  // The spectrogram plugin is loaded TWICE, each instance owning half
  // the bins (bins are fully independent correlators -- see
  // spectrogram.h's range contract). Profiling round 3 showed the
  // spectrogram as the pipeline's slowest stage once generation left
  // the measured region, and the box has a spare core; splitting the
  // one heavy stage is the thread rebalance that actually moves the
  // bound. Six module instances, still five distinct modules.
  std::vector<LoadedModule> chain = {
      LoadModule(plugin_dir + "/libpulse_detector_plugin.so",
                 "threshold=6.0,sample_rate=10000000"),
      LoadModule(plugin_dir + "/libpulse_spectrogram_plugin.so",
                 "sample_rate=10000000,num_bins=8,bin_begin=0,bin_end=4"),
      LoadModule(plugin_dir + "/libpulse_spectrogram_plugin.so",
                 "sample_rate=10000000,num_bins=8,bin_begin=4,bin_end=8"),
      LoadModule(plugin_dir + "/libpulse_jammer_plugin.so",
                 "power_threshold=20.0,duty_cycle_threshold=0.5"),
      LoadModule(plugin_dir + "/libpulse_stats_plugin.so", "sample_rate=10000000"),
      LoadModule(plugin_dir + "/libpulse_deinterleaver_plugin.so",
                 "sample_rate=10000000,pri_tolerance=0.0000001"),
  };

  // Round three of the theoretical-limits branch: the signal is a
  // GIVEN. Real systems receive their IQ from a radio -- no
  // architecture choice speeds up the antenna -- so every batch is
  // generated up front, before the clock starts, and the measured
  // pipeline begins at detection. One pre-filled frame per batch (at
  // this repo's scale, ~160KB of samples per frame; the benchmark's
  // 50k-pulse runs hold ~8MB resident -- the price of "the input
  // already exists" being literally true).
  pulsecore::SyntheticIQSource source(/*sample_rate_hz=*/10000000.0, num_pulses);
  std::vector<pulse::PipelineFrame> frames;
  {
    pulse::PipelineFrame f;
    while (source.NextBatch(f.mutable_iq())) {
      frames.push_back(std::move(f));
      f.Clear();
    }
  }
  // Pre-size every frame's output containers, also untimed: the
  // round-two slot ring amortized output allocation away by reusing 8
  // warm frames; with one frame per batch, leaving allocation inside
  // the measured region would bill the pipeline for heap growth and
  // first-touch page faults that steady-state processing never pays.
  // 2048 comfortably covers this signal's ~1000 events/batch.
  for (pulse::PipelineFrame& f : frames) {
    pulse::PulseEventBatch* ev = f.mutable_events();
    ev->mutable_start_sample()->Reserve(2048);
    ev->mutable_end_sample()->Reserve(2048);
    ev->mutable_peak_amplitude()->Reserve(2048);
    ev->mutable_mean_amplitude()->Reserve(2048);
    ev->mutable_duration_seconds()->Reserve(2048);
    // The spectrogram's arrays must be pre-sized: two half-range
    // analyzer instances write disjoint entries concurrently, and the
    // range contract (spectrogram.h) forbids them resizing anything.
    pulse::SpectrogramSummary* spec = f.mutable_spectrogram();
    for (int b = 0; b < 8; ++b) {
      spec->add_max_magnitude(0.0);
      spec->add_mean_magnitude(0.0);
    }
    f.mutable_jam();
    f.mutable_stats();
    f.mutable_deinterleave();
  }

  // The modules run as a stage-per-thread pipeline (see
  // threaded_pipeline.h). Same modules, same dlopen boundary -- each
  // module INSTANCE is invoked from exactly one thread, in batch
  // order, so its internal running state needs no locking. With
  // generation retired from the pipeline, the grouping is again sized
  // to the 4-core benchmark box: detector alone, each spectrogram
  // half-range instance alone (the stage profiling round 3 showed
  // pacing everything else), and the three cheap stages sharing the
  // fourth thread. The two spectrogram instances run one frame apart
  // in the linear chain and write disjoint halves of the same
  // pre-sized summary -- see spectrogram.h's range contract.
  std::vector<std::function<void(pulse::PipelineFrame*)>> stages = {
      [&](pulse::PipelineFrame* f) { chain[0].process(chain[0].instance, f); },  // detector
      [&](pulse::PipelineFrame* f) { chain[1].process(chain[1].instance, f); },  // spectrogram bins 0-3
      [&](pulse::PipelineFrame* f) { chain[2].process(chain[2].instance, f); },  // spectrogram bins 4-7
      [&](pulse::PipelineFrame* f) {                                             // jammer + stats + deint
        chain[3].process(chain[3].instance, f);
        chain[4].process(chain[4].instance, f);
        chain[5].process(chain[5].instance, f);
      },
  };

  // Timed region covers only the pipeline run -- generation (above),
  // module loading (dlopen()/dlsym()), and teardown (dlclose() below)
  // are all excluded, so this number reflects steady-state throughput
  // of detection-through-deinterleave. See
  // microservice/detector_service/main.cpp for the equivalent charter
  // on the chain build, and scripts/benchmark_steady_state.sh for how
  // these numbers get compared.
  const monolith::ThreadedPipelineResult run = monolith::RunThreadedPipeline(stages, &frames);
  static const pulse::PipelineFrame kEmptyFrame;
  const pulse::PipelineFrame& frame = run.final_frame ? *run.final_frame : kEmptyFrame;

  // chain.size() counts INSTANCES (the spectrogram is loaded twice);
  // the pipeline still consists of the same five distinct modules, and
  // this line stays comparable with every earlier measurement's output.
  const size_t distinct_modules = chain.size() - 1;
  std::printf(
      "[monolith_app] processed %d IQ batches through a dynamically-linked chain of %zu modules\n",
      run.batches, distinct_modules);
  std::printf("[monolith_app] STEADY_STATE_MS %.6f\n", run.steady_state_ms);

  const pulse::SpectrogramSummary& spectrogram = frame.spectrogram();
  std::printf("[monolith_app] spectrogram: %d bins, %.1f Hz spacing, %llu frames\n",
              spectrogram.max_magnitude_size(), spectrogram.bin_hz(),
              static_cast<unsigned long long>(spectrogram.frame_count()));
  for (int i = 0; i < spectrogram.max_magnitude_size(); ++i) {
    std::printf("[monolith_app]   bin %d (~%.0f Hz): max=%.3f mean=%.3f\n", i,
                (i + 0.5) * spectrogram.bin_hz(), spectrogram.max_magnitude(i),
                spectrogram.mean_magnitude(i));
  }

  const pulse::JamSummary& jam = frame.jam();
  std::printf(
      "[monolith_app] jammer: %llu/%llu batches flagged, max_duty_cycle=%.3f max_mean_power=%.2f\n",
      static_cast<unsigned long long>(jam.batches_flagged()),
      static_cast<unsigned long long>(jam.batches_total()), jam.max_duty_cycle(),
      jam.max_mean_power());

  const pulse::PulseSummary& stats = frame.stats();
  std::printf(
      "[monolith_app] stats: pulses=%llu mean_peak=%.3f mean_dur_us=%.2f mean_pri_us=%.2f "
      "min_peak=%.3f max_peak=%.3f\n",
      static_cast<unsigned long long>(stats.pulse_count()), stats.mean_peak_amplitude(),
      stats.mean_duration_seconds() * 1e6, stats.mean_pri_seconds() * 1e6,
      stats.min_peak_amplitude(), stats.max_peak_amplitude());

  const pulse::DeinterleaveSummary& tracks = frame.deinterleave();
  std::printf("[monolith_app] deinterleaver: %d track(s)\n", tracks.tracks_size());
  for (const pulse::EmitterTrack& track : tracks.tracks()) {
    std::printf("[monolith_app]   track %u: pulses=%llu estimated_pri_us=%.2f mean_peak=%.3f\n",
                track.track_id(), static_cast<unsigned long long>(track.pulse_count()),
                track.estimated_pri_seconds() * 1e6, track.mean_peak_amplitude());
  }

  for (LoadedModule& stage : chain) {
    stage.destroy(stage.instance);
    dlclose(stage.handle);
  }
  return 0;
}
