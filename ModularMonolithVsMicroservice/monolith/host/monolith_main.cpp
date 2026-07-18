// monolith_app: a single process/executable that dlopen()s five
// pulse-processing modules at startup and wires them together with
// plain C++ references -- no serialization, since every module is built
// from the same pulse.pb.h/pulse_proto as the host (see module_api.h for
// why that makes it safe). Two modules (spectrogram, jammer) consume the
// raw IQBatch directly, same as the detector; two more (stats,
// deinterleaver) consume the detector's PulseEventBatch output. Compare
// with microservice/{detector_service,stats_service,spectrogram_service,
// jammer_service,deinterleave_service}, which run the identical core
// algorithms as five separate processes that must serialize onto TCP
// sockets because they don't share an address space.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "iq_source.h"
#include "module_api.h"
#include "pulse.pb.h"

namespace {

// ProcessFn differs between module kinds (they consume/produce different
// message types), so this is templated on it rather than sharing one
// non-generic struct/loader between all five.
template <typename ProcessFn>
struct LoadedModule {
  void* handle = nullptr;
  pulse_module_create_fn create = nullptr;
  pulse_module_destroy_fn destroy = nullptr;
  ProcessFn process = nullptr;
  pulse_module_t instance = nullptr;
};

template <typename ProcessFn>
LoadedModule<ProcessFn> LoadModule(const std::string& path, const char* process_symbol,
                                    const char* config) {
  LoadedModule<ProcessFn> m;
  m.handle = dlopen(path.c_str(), RTLD_NOW);
  if (m.handle == nullptr) {
    std::fprintf(stderr, "[monolith_app] dlopen(%s) failed: %s\n", path.c_str(), dlerror());
    std::exit(1);
  }

  m.create = reinterpret_cast<pulse_module_create_fn>(dlsym(m.handle, PULSE_MODULE_CREATE_SYM));
  m.destroy = reinterpret_cast<pulse_module_destroy_fn>(dlsym(m.handle, PULSE_MODULE_DESTROY_SYM));
  m.process = reinterpret_cast<ProcessFn>(dlsym(m.handle, process_symbol));
  if (m.create == nullptr || m.destroy == nullptr || m.process == nullptr) {
    std::fprintf(stderr, "[monolith_app] %s is missing a required symbol\n", path.c_str());
    std::exit(1);
  }

  m.instance = m.create(config);
  std::printf("[monolith_app] loaded %s\n", path.c_str());
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string plugin_dir = argc > 1 ? argv[1] : ".";
  const int num_pulses = argc > 2 ? std::atoi(argv[2]) : 6;

  LoadedModule<pulse_detector_process_fn> detector =
      LoadModule<pulse_detector_process_fn>(plugin_dir + "/libpulse_detector_plugin.so",
                                             PULSE_DETECTOR_PROCESS_SYM,
                                             "threshold=6.0,sample_rate=1000000");
  LoadedModule<pulse_stats_process_fn> stats = LoadModule<pulse_stats_process_fn>(
      plugin_dir + "/libpulse_stats_plugin.so", PULSE_STATS_PROCESS_SYM, "sample_rate=1000000");
  LoadedModule<pulse_spectrogram_process_fn> spectrogram =
      LoadModule<pulse_spectrogram_process_fn>(plugin_dir + "/libpulse_spectrogram_plugin.so",
                                                PULSE_SPECTROGRAM_PROCESS_SYM,
                                                "sample_rate=1000000,num_bins=8");
  LoadedModule<pulse_jammer_process_fn> jammer = LoadModule<pulse_jammer_process_fn>(
      plugin_dir + "/libpulse_jammer_plugin.so", PULSE_JAMMER_PROCESS_SYM,
      "power_threshold=20.0,duty_cycle_threshold=0.5");
  LoadedModule<pulse_deinterleaver_process_fn> deinterleaver =
      LoadModule<pulse_deinterleaver_process_fn>(plugin_dir + "/libpulse_deinterleaver_plugin.so",
                                                  PULSE_DEINTERLEAVER_PROCESS_SYM,
                                                  "sample_rate=1000000,pri_tolerance=0.000005");

  pulsecore::SyntheticIQSource source(/*sample_rate_hz=*/1000000.0, num_pulses);
  pulse::IQBatch iq_batch;
  // Reused across iterations for the same reason common/ loops reuse
  // their message objects -- see pulse_detector_plugin.cpp's history.
  pulse::PulseEventBatch events;
  pulse::PulseSummary last_summary;
  pulse::SpectrogramSummary last_spectrogram;
  pulse::JamSummary last_jam;
  pulse::DeinterleaveSummary last_tracks;
  int batches = 0;

  while (source.NextBatch(&iq_batch)) {
    // Straight in-process calls: everything is passed by reference into
    // the dlopen()'d modules and read/written in place. spectrogram and
    // jammer consume the raw batch directly, same as detector; stats and
    // deinterleaver consume the detector's derived events.
    detector.process(detector.instance, iq_batch, &events);
    stats.process(stats.instance, events, &last_summary);
    spectrogram.process(spectrogram.instance, iq_batch, &last_spectrogram);
    jammer.process(jammer.instance, iq_batch, &last_jam);
    deinterleaver.process(deinterleaver.instance, events, &last_tracks);
    ++batches;
  }

  std::printf("[monolith_app] processed %d IQ batches through dynamically-linked modules\n",
              batches);
  std::printf(
      "[monolith_app] stats: pulses=%llu mean_peak=%.3f mean_dur_us=%.2f mean_pri_us=%.2f "
      "min_peak=%.3f max_peak=%.3f\n",
      static_cast<unsigned long long>(last_summary.pulse_count()),
      last_summary.mean_peak_amplitude(), last_summary.mean_duration_seconds() * 1e6,
      last_summary.mean_pri_seconds() * 1e6, last_summary.min_peak_amplitude(),
      last_summary.max_peak_amplitude());

  std::printf("[monolith_app] spectrogram: %d bins, %.1f Hz spacing, %llu frames\n",
              last_spectrogram.max_magnitude_size(), last_spectrogram.bin_hz(),
              static_cast<unsigned long long>(last_spectrogram.frame_count()));
  for (int i = 0; i < last_spectrogram.max_magnitude_size(); ++i) {
    std::printf("[monolith_app]   bin %d (~%.0f Hz): max=%.3f mean=%.3f\n", i,
                (i + 0.5) * last_spectrogram.bin_hz(), last_spectrogram.max_magnitude(i),
                last_spectrogram.mean_magnitude(i));
  }

  std::printf(
      "[monolith_app] jammer: %llu/%llu batches flagged, max_duty_cycle=%.3f max_mean_power=%.2f\n",
      static_cast<unsigned long long>(last_jam.batches_flagged()),
      static_cast<unsigned long long>(last_jam.batches_total()), last_jam.max_duty_cycle(),
      last_jam.max_mean_power());

  std::printf("[monolith_app] deinterleaver: %d track(s)\n", last_tracks.tracks_size());
  for (const pulse::EmitterTrack& track : last_tracks.tracks()) {
    std::printf("[monolith_app]   track %u: pulses=%llu estimated_pri_us=%.2f mean_peak=%.3f\n",
                track.track_id(), static_cast<unsigned long long>(track.pulse_count()),
                track.estimated_pri_seconds() * 1e6, track.mean_peak_amplitude());
  }

  detector.destroy(detector.instance);
  stats.destroy(stats.instance);
  spectrogram.destroy(spectrogram.instance);
  jammer.destroy(jammer.instance);
  deinterleaver.destroy(deinterleaver.instance);
  dlclose(detector.handle);
  dlclose(stats.handle);
  dlclose(spectrogram.handle);
  dlclose(jammer.handle);
  dlclose(deinterleaver.handle);
  return 0;
}
