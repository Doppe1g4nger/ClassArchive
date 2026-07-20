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
  SpectrogramModule(double sample_rate_hz, int num_bins, int bin_begin, int bin_end)
      : analyzer(sample_rate_hz, num_bins, bin_begin, bin_end) {}
  pulsecore::SpectrogramAnalyzer analyzer;
};

}  // namespace

extern "C" {

pulse_module_t pulse_module_create(const char* config) {
  double sample_rate_hz = 10000000.0;
  int num_bins = 8;
  // Optional bin range (theoretical-limits branch): lets the host load
  // this plugin twice and split the stage's independent bins across
  // two threads -- see spectrogram.h's range/output contract. Absent
  // (the default), one instance owns all bins, exactly as before.
  int bin_begin = -1;
  int bin_end = -1;
  if (config != nullptr) {
    if (std::sscanf(config, "sample_rate=%lf,num_bins=%d,bin_begin=%d,bin_end=%d",
                    &sample_rate_hz, &num_bins, &bin_begin, &bin_end) < 4) {
      bin_begin = -1;
      bin_end = -1;
      std::sscanf(config, "sample_rate=%lf,num_bins=%d", &sample_rate_hz, &num_bins);
    }
  }
  return new SpectrogramModule(sample_rate_hz, num_bins, bin_begin, bin_end);
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
