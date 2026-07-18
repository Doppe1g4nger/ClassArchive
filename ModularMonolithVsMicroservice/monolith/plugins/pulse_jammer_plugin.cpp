// Built into libpulse_jammer_plugin.so and dlopen()'d by monolith_app.
// Third stage of the pipeline chain (detector -> spectrogram -> jammer ->
// stats -> deinterleaver) -- the last of the three stages that read
// frame.iq(), which is why it's grouped here instead of at the end: the
// microservice build clears frame.iq() right after this stage runs (see
// microservice/jammer_service/main.cpp), since nothing downstream needs
// it. Wraps pulsecore::JammerDetector.

#include <cstdio>

#include "jammer.h"
#include "module_api.h"
#include "pulse.pb.h"

namespace {

struct JammerModule {
  JammerModule(double power_threshold, double duty_cycle_threshold)
      : detector(power_threshold, duty_cycle_threshold) {}
  pulsecore::JammerDetector detector;
};

}  // namespace

extern "C" {

pulse_module_t pulse_module_create(const char* config) {
  double power_threshold = 20.0;
  double duty_cycle_threshold = 0.5;
  if (config != nullptr) {
    std::sscanf(config, "power_threshold=%lf,duty_cycle_threshold=%lf", &power_threshold,
                &duty_cycle_threshold);
  }
  return new JammerModule(power_threshold, duty_cycle_threshold);
}

void pulse_module_destroy(pulse_module_t handle) {
  delete static_cast<JammerModule*>(handle);
}

// Reads frame->iq (populated once, before the chain starts) and writes
// frame->jam.
void pulse_stage_process(pulse_module_t handle, pulse::PipelineFrame* frame) {
  auto* module = static_cast<JammerModule*>(handle);
  module->detector.Process(frame->iq(), frame->mutable_jam());
}

}  // extern "C"
