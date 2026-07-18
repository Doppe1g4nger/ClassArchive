// Built into libpulse_spectrogram_plugin.so and dlopen()'d by
// monolith_app. Wraps pulsecore::SpectrogramAnalyzer behind the typed
// C++ ABI defined in module_api.h.

#include <cstdio>

#include "module_api.h"
#include "pulse.pb.h"
#include "spectrogram.h"

namespace {

struct SpectrogramModule {
  SpectrogramModule(double sample_rate_hz, int num_bins) : analyzer(sample_rate_hz, num_bins) {}
  pulsecore::SpectrogramAnalyzer analyzer;
};

}  // namespace

extern "C" {

pulse_module_t pulse_module_create(const char* config) {
  double sample_rate_hz = 1000000.0;
  int num_bins = 8;
  if (config != nullptr) {
    std::sscanf(config, "sample_rate=%lf,num_bins=%d", &sample_rate_hz, &num_bins);
  }
  return new SpectrogramModule(sample_rate_hz, num_bins);
}

void pulse_module_destroy(pulse_module_t handle) {
  delete static_cast<SpectrogramModule*>(handle);
}

void pulse_spectrogram_process(pulse_module_t handle, const pulse::IQBatch& batch,
                                pulse::SpectrogramSummary* out) {
  auto* module = static_cast<SpectrogramModule*>(handle);
  module->analyzer.Process(batch, out);
}

}  // extern "C"
