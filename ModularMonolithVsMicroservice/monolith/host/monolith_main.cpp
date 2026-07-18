// monolith_app: a single process/executable that dlopen()s five
// pulse-processing modules at startup and runs them as a linear chain --
// detector -> stats -> deinterleaver -> spectrogram -> jammer -- each
// stage reading/writing one pulse::PipelineFrame passed by reference, no
// serialization, since every module is built from the same
// pulse.pb.h/pulse_proto as the host (see module_api.h for why that makes
// it safe). Compare with microservice/{detector_service,stats_service,
// deinterleave_service,spectrogram_service,jammer_service}, which run the
// identical core algorithms as five separate processes chained over TCP
// sockets -- each one both a server to the stage before it and a client
// to the stage after it -- because they don't share an address space.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "iq_source.h"
#include "module_api.h"
#include "pulse.pb.h"

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
  const int num_pulses = argc > 2 ? std::atoi(argv[2]) : 6;

  // Chain order matches microservice/'s wiring exactly (see
  // scripts/run_microservices.sh): each stage needs whatever the stages
  // before it in this list have already written into the frame.
  std::vector<LoadedModule> chain = {
      LoadModule(plugin_dir + "/libpulse_detector_plugin.so", "threshold=6.0,sample_rate=1000000"),
      LoadModule(plugin_dir + "/libpulse_stats_plugin.so", "sample_rate=1000000"),
      LoadModule(plugin_dir + "/libpulse_deinterleaver_plugin.so",
                 "sample_rate=1000000,pri_tolerance=0.000005"),
      LoadModule(plugin_dir + "/libpulse_spectrogram_plugin.so", "sample_rate=1000000,num_bins=8"),
      LoadModule(plugin_dir + "/libpulse_jammer_plugin.so",
                 "power_threshold=20.0,duty_cycle_threshold=0.5"),
  };

  pulsecore::SyntheticIQSource source(/*sample_rate_hz=*/1000000.0, num_pulses);
  // Reused across iterations for the same reason common/ loops reuse
  // their message objects -- see pulse_detector_plugin.cpp's history.
  // frame.iq() is filled directly by NextBatch() below (no copy); every
  // other field is written by whichever chain stage owns it.
  pulse::PipelineFrame frame;
  int batches = 0;

  while (source.NextBatch(frame.mutable_iq())) {
    for (LoadedModule& stage : chain) {
      stage.process(stage.instance, &frame);
    }
    ++batches;
  }

  std::printf("[monolith_app] processed %d IQ batches through a dynamically-linked chain of %zu modules\n",
              batches, chain.size());

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

  for (LoadedModule& stage : chain) {
    stage.destroy(stage.instance);
    dlclose(stage.handle);
  }
  return 0;
}
