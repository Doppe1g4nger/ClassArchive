// Built into libpulse_stats_plugin.so and dlopen()'d by monolith_app.
// Fourth stage of the pipeline chain (detector -> spectrogram -> jammer
// -> stats -> deinterleaver). Wraps pulsecore::PulseStatsAccumulator.

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
  double sample_rate_hz = 10000000.0;
  if (config != nullptr) {
    std::sscanf(config, "sample_rate=%lf", &sample_rate_hz);
  }
  return new StatsModule(sample_rate_hz);
}

void pulse_module_destroy(pulse_module_t handle) {
  delete static_cast<StatsModule*>(handle);
}

// Reads frame->events (populated by the detector stage) and writes
// frame->stats.
void pulse_stage_process(pulse_module_t handle, pulse::PipelineFrame* frame) {
  auto* module = static_cast<StatsModule*>(handle);
  module->accumulator.Add(frame->events());
  *frame->mutable_stats() = module->accumulator.Finalize();
}

}  // extern "C"
