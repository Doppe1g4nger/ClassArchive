// Built into libpulse_spectrogram_plugin.so and dlopen()'d by
// monolith_app. Second stage of the pipeline chain (detector ->
// spectrogram -> jammer -> stats -> deinterleaver). Wraps
// pulsecore::SpectrogramAnalyzer.

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
  double sample_rate_hz = 10000000.0;
  int num_bins = 8;
  if (config != nullptr) {
    std::sscanf(config, "sample_rate=%lf,num_bins=%d", &sample_rate_hz, &num_bins);
  }
  return new SpectrogramModule(sample_rate_hz, num_bins);
}

void pulse_module_destroy(pulse_module_t handle) {
  delete static_cast<SpectrogramModule*>(handle);
}

// Reads frame->iq (populated once, before the chain starts) and writes
// frame->spectrogram.
void pulse_stage_process(pulse_module_t handle, pulse::PipelineFrame* frame) {
  auto* module = static_cast<SpectrogramModule*>(handle);
  module->analyzer.Process(frame->iq(), frame->mutable_spectrogram());
}

}  // extern "C"
