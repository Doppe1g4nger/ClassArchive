// monolith_app: a single process/executable that dlopen()s both the
// pulse-detector and pulse-stats modules at startup and wires the
// detector's output directly into the stats module's input as a plain
// C++ reference -- no serialization, since both modules are built from
// the same pulse.pb.h/pulse_proto as the host (see module_api.h for why
// that makes it safe). Compare with
// microservice/{detector_service,stats_service}, which run the identical
// core algorithms as two separate processes that must serialize onto a
// TCP socket because they don't share an address space.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "iq_source.h"
#include "module_api.h"
#include "pulse.pb.h"

namespace {

// ProcessFn differs between the detector and stats modules (they consume
// and produce different message types), so this is templated on it
// rather than sharing one non-generic struct/loader between the two.
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

  pulsecore::SyntheticIQSource source(/*sample_rate_hz=*/1000000.0, num_pulses);
  pulse::IQBatch iq_batch;
  // Reused across iterations for the same reason common/ loops reuse
  // their message objects -- see pulse_detector_plugin.cpp's history.
  pulse::PulseEventBatch events;
  pulse::PulseSummary last_summary;
  int batches = 0;

  while (source.NextBatch(&iq_batch)) {
    // Straight in-process calls: iq_batch and events are passed by
    // reference into the dlopen()'d modules and read/written in place.
    detector.process(detector.instance, iq_batch, &events);
    stats.process(stats.instance, events, &last_summary);
    ++batches;
  }

  std::printf("[monolith_app] processed %d IQ batches through dynamically-linked modules\n",
              batches);
  std::printf(
      "[monolith_app] pulses=%llu mean_peak=%.3f mean_dur_us=%.2f mean_pri_us=%.2f "
      "min_peak=%.3f max_peak=%.3f\n",
      static_cast<unsigned long long>(last_summary.pulse_count()),
      last_summary.mean_peak_amplitude(), last_summary.mean_duration_seconds() * 1e6,
      last_summary.mean_pri_seconds() * 1e6, last_summary.min_peak_amplitude(),
      last_summary.max_peak_amplitude());

  detector.destroy(detector.instance);
  stats.destroy(stats.instance);
  dlclose(detector.handle);
  dlclose(stats.handle);
  return 0;
}
