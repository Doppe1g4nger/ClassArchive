// Built into libpulse_stats_plugin.so and dlopen()'d by monolith_app.
// Wraps pulsecore::PulseStatsAccumulator behind the C ABI defined in
// module_api.h. Each call consumes one PulseEventBatch and returns the
// running PulseSummary computed so far.

#include <cstdio>

#include "module_api.h"
#include "pulse.pb.h"
#include "pulse_stats.h"

namespace {

struct StatsModule {
  explicit StatsModule(double sample_rate_hz) : accumulator(sample_rate_hz) {}
  pulsecore::PulseStatsAccumulator accumulator;

  // Reused across process() calls; see the equivalent member in
  // pulse_detector_plugin.cpp for why.
  pulse::PulseEventBatch in_batch;
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

int pulse_module_process(pulse_module_t handle, const uint8_t* in_bytes, uint32_t in_len,
                          uint8_t** out_bytes, uint32_t* out_len) {
  auto* module = static_cast<StatsModule*>(handle);

  if (!module->in_batch.ParseFromArray(in_bytes, static_cast<int>(in_len))) {
    return -1;
  }
  module->accumulator.Add(module->in_batch);

  const pulse::PulseSummary summary = module->accumulator.Finalize();
  const uint32_t size = static_cast<uint32_t>(summary.ByteSizeLong());
  uint8_t* buffer = new uint8_t[size];
  if (!summary.SerializeToArray(buffer, static_cast<int>(size))) {
    delete[] buffer;
    return -1;
  }

  *out_bytes = buffer;
  *out_len = size;
  return 0;
}

void pulse_module_free_buffer(uint8_t* buffer) { delete[] buffer; }

}  // extern "C"
