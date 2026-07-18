// Built into libpulse_detector_plugin.so and dlopen()'d by monolith_app.
// Wraps pulsecore::PulseDetector behind the C ABI defined in module_api.h.

#include <cstdio>
#include <cstring>

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

int pulse_module_process(pulse_module_t handle, const uint8_t* in_bytes, uint32_t in_len,
                          uint8_t** out_bytes, uint32_t* out_len) {
  auto* module = static_cast<DetectorModule*>(handle);

  pulse::IQBatch batch;
  if (!batch.ParseFromArray(in_bytes, static_cast<int>(in_len))) {
    return -1;
  }

  pulse::PulseEventBatch events;
  module->detector.Process(batch, &events);

  const uint32_t size = static_cast<uint32_t>(events.ByteSizeLong());
  uint8_t* buffer = new uint8_t[size];
  if (!events.SerializeToArray(buffer, static_cast<int>(size))) {
    delete[] buffer;
    return -1;
  }

  *out_bytes = buffer;
  *out_len = size;
  return 0;
}

void pulse_module_free_buffer(uint8_t* buffer) { delete[] buffer; }

}  // extern "C"
