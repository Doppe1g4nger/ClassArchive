// Built into libpulse_detector_plugin.so and dlopen()'d by monolith_app.
// Wraps pulsecore::PulseDetector behind the typed C++ ABI defined in
// module_api.h -- no serialization, since this module and its host share
// one in-process definition of every pulse:: type (see module_api.h).

#include <cstdio>

#include "module_api.h"
#include "pulse.pb.h"
#include "pulse_detector.h"

namespace {

struct DetectorModule {
  explicit DetectorModule(double threshold, double sample_rate_hz)
      : detector(threshold, sample_rate_hz) {}
  pulsecore::PulseDetector detector;
};

}  // namespace

extern "C" {

pulse_module_t pulse_module_create(const char* config) {
  double threshold = 6.0;
  double sample_rate_hz = 1000000.0;
  if (config != nullptr) {
    // Deliberately minimal config format ("threshold=6.0,sample_rate=1e6")
    // to avoid pulling in a JSON dependency for a two-field config.
    std::sscanf(config, "threshold=%lf,sample_rate=%lf", &threshold, &sample_rate_hz);
  }
  return new DetectorModule(threshold, sample_rate_hz);
}

void pulse_module_destroy(pulse_module_t handle) {
  delete static_cast<DetectorModule*>(handle);
}

void pulse_detector_process(pulse_module_t handle, const pulse::IQBatch& batch,
                             pulse::PulseEventBatch* out) {
  auto* module = static_cast<DetectorModule*>(handle);
  out->Clear();
  module->detector.Process(batch, out);
}

}  // extern "C"
