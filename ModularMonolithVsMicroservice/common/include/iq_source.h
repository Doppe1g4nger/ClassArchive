#pragma once

#include <cstdint>

#include "pulse.pb.h"

namespace pulsecore {

// Deterministic synthetic IQ generator: emits a handful of rectangular
// pulses of known amplitude buried in low-level noise, batch by batch.
// Its only job is to give both the monolith and the microservice builds
// byte-for-byte identical input so their output can be compared directly.
class SyntheticIQSource {
 public:
  SyntheticIQSource(double sample_rate_hz, int num_pulses, uint32_t seed = 42);

  // Fills `batch` with the next chunk of samples. Returns false once every
  // requested pulse has been emitted and there is nothing left to produce.
  bool NextBatch(pulse::IQBatch* batch);

 private:
  double NextNoise();

  double sample_rate_hz_;
  int num_pulses_;
  uint64_t total_samples_;
  uint64_t sample_cursor_ = 0;
  uint32_t rng_state_;

  static constexpr int kBatchSize = 256;
  static constexpr int kGapSamples = 400;
  static constexpr int kPulseSamples = 120;
  static constexpr double kPulseAmplitude = 10.0;
  static constexpr double kNoiseAmplitude = 0.5;
};

}  // namespace pulsecore
