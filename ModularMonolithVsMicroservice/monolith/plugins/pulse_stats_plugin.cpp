// Built into libpulse_stats_plugin.so and dlopen()'d by monolith_app.
// Wraps pulsecore::PulseStatsAccumulator behind the typed C++ ABI defined
// in module_api.h. Each call folds one PulseEventBatch into running state
// and writes the summary computed so far into *out.

#include <cstdio>

#include "module_api.h"
#include "pulse.pb.h"
#include "pulse_stats.h"

namespace {

struct StatsModule {
  explicit StatsModule(double sample_rate_hz) : accumulator(sample_rate_hz) {}
  pulsecore::PulseStatsAccumulator accumulator;
};

}  // namespace

extern "C" {

pulse_module_t pulse_module_create(const char* config) {
  double sample_rate_hz = 1000000.0;
  if (config != nullptr) {
    std::sscanf(config, "sample_rate=%lf", &sample_rate_hz);
  }
  return new StatsModule(sample_rate_hz);
}

void pulse_module_destroy(pulse_module_t handle) {
  delete static_cast<StatsModule*>(handle);
}

void pulse_stats_process(pulse_module_t handle, const pulse::PulseEventBatch& batch,
                          pulse::PulseSummary* out) {
  auto* module = static_cast<StatsModule*>(handle);
  module->accumulator.Add(batch);
  *out = module->accumulator.Finalize();
}

}  // extern "C"
