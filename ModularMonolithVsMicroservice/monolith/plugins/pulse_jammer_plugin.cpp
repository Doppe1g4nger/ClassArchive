// Built into libpulse_jammer_plugin.so and dlopen()'d by monolith_app.
// Wraps pulsecore::JammerDetector behind the typed C++ ABI defined in
// module_api.h.

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

void pulse_jammer_process(pulse_module_t handle, const pulse::IQBatch& batch,
                           pulse::JamSummary* out) {
  auto* module = static_cast<JammerModule*>(handle);
  module->detector.Process(batch, out);
}

}  // extern "C"
