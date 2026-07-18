// Built into libpulse_deinterleaver_plugin.so and dlopen()'d by
// monolith_app. Fifth and final stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver). Wraps
// pulsecore::Deinterleaver.

#include <cstdio>

#include "deinterleaver.h"
#include "module_api.h"
#include "pulse.pb.h"

namespace {

struct DeinterleaverModule {
  DeinterleaverModule(double sample_rate_hz, double pri_tolerance_seconds)
      : deinterleaver(sample_rate_hz, pri_tolerance_seconds) {}
  pulsecore::Deinterleaver deinterleaver;
};

}  // namespace

extern "C" {

pulse_module_t pulse_module_create(const char* config) {
  double sample_rate_hz = 1000000.0;
  double pri_tolerance_seconds = 5e-6;
  if (config != nullptr) {
    std::sscanf(config, "sample_rate=%lf,pri_tolerance=%lf", &sample_rate_hz,
                &pri_tolerance_seconds);
  }
  return new DeinterleaverModule(sample_rate_hz, pri_tolerance_seconds);
}

void pulse_module_destroy(pulse_module_t handle) {
  delete static_cast<DeinterleaverModule*>(handle);
}

// Reads frame->events (populated by the detector stage) and writes
// frame->deinterleave.
void pulse_stage_process(pulse_module_t handle, pulse::PipelineFrame* frame) {
  auto* module = static_cast<DeinterleaverModule*>(handle);
  module->deinterleaver.Process(frame->events(), frame->mutable_deinterleave());
}

}  // extern "C"
