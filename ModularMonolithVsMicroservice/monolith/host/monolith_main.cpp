// monolith_app: a single process/executable that dlopen()s both the
// pulse-detector and pulse-stats modules at startup and wires the
// detector's protobuf output directly into the stats module's input via
// plain function calls. No sockets, no serialization framing -- just
// bytes handed across a dlopen() boundary. Compare with
// microservice/{detector_service,stats_service} which run the identical
// core algorithms as two separate processes talking protobuf over TCP.

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "iq_source.h"
#include "module_api.h"
#include "pulse.pb.h"

namespace {

struct LoadedModule {
  void* handle = nullptr;
  pulse_module_create_fn create = nullptr;
  pulse_module_destroy_fn destroy = nullptr;
  pulse_module_process_fn process = nullptr;
  pulse_module_free_buffer_fn free_buffer = nullptr;
  pulse_module_t instance = nullptr;
};

LoadedModule LoadModule(const std::string& path, const char* config) {
  LoadedModule m;
  m.handle = dlopen(path.c_str(), RTLD_NOW);
  if (m.handle == nullptr) {
    std::fprintf(stderr, "[monolith_app] dlopen(%s) failed: %s\n", path.c_str(), dlerror());
    std::exit(1);
  }

  m.create = reinterpret_cast<pulse_module_create_fn>(dlsym(m.handle, PULSE_MODULE_CREATE_SYM));
  m.destroy = reinterpret_cast<pulse_module_destroy_fn>(dlsym(m.handle, PULSE_MODULE_DESTROY_SYM));
  m.process = reinterpret_cast<pulse_module_process_fn>(dlsym(m.handle, PULSE_MODULE_PROCESS_SYM));
  m.free_buffer =
      reinterpret_cast<pulse_module_free_buffer_fn>(dlsym(m.handle, PULSE_MODULE_FREE_BUFFER_SYM));
  if (m.create == nullptr || m.destroy == nullptr || m.process == nullptr ||
      m.free_buffer == nullptr) {
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

  LoadedModule detector =
      LoadModule(plugin_dir + "/libpulse_detector_plugin.so", "threshold=6.0,sample_rate=1000000");
  LoadedModule stats = LoadModule(plugin_dir + "/libpulse_stats_plugin.so", "sample_rate=1000000");

  pulsecore::SyntheticIQSource source(/*sample_rate_hz=*/1000000.0, /*num_pulses=*/6);
  pulse::IQBatch iq_batch;
  pulse::PulseSummary last_summary;
  int batches = 0;

  while (source.NextBatch(&iq_batch)) {
    std::string in_bytes;
    iq_batch.SerializeToString(&in_bytes);

    uint8_t* det_out = nullptr;
    uint32_t det_out_len = 0;
    if (detector.process(detector.instance, reinterpret_cast<const uint8_t*>(in_bytes.data()),
                          static_cast<uint32_t>(in_bytes.size()), &det_out, &det_out_len) != 0) {
      std::fprintf(stderr, "[monolith_app] detector module failed\n");
      return 1;
    }

    uint8_t* stats_out = nullptr;
    uint32_t stats_out_len = 0;
    const int rc = stats.process(stats.instance, det_out, det_out_len, &stats_out, &stats_out_len);
    detector.free_buffer(det_out);
    if (rc != 0) {
      std::fprintf(stderr, "[monolith_app] stats module failed\n");
      return 1;
    }

    last_summary.ParseFromArray(stats_out, static_cast<int>(stats_out_len));
    stats.free_buffer(stats_out);
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
