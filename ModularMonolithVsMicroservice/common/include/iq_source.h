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

  // Tuned for a 1,000,000-pulse-per-second pulse train (PRF = 1 MHz) split
  // into 1000-microsecond (1ms) buffers, which at the 10,000,000 Hz
  // (10 MSps) sample rate every caller in this repo constructs this class
  // with (see monolith_main.cpp / the microservice *_service mains) works
  // out to a clean whole-sample period:
  //   period = kGapSamples + kPulseSamples = 10 samples = 1 microsecond
  //          -> PRF = sample_rate_hz / period = 1,000,000 pulses/sec
  //   kBatchSize = 10,000 samples = 1000 microseconds (1ms) of signal
  //          -> 1000 pulses per batch, exactly
  // kPulseSamples:kGapSamples keeps roughly the original 23%-ish duty
  // cycle (2:8 = 20%) so the jammer's duty-cycle threshold (tuned against
  // that ratio -- see common/include/jammer.h) still doesn't misfire on
  // ordinary pulsed traffic. Changing sample_rate_hz at a call site
  // without changing these breaks the "exactly 1000 pulses/batch" property
  // (it's this class's job to keep that property true only for the one
  // sample rate documented above, not to auto-derive it from the rate
  // actually passed in).
  static constexpr int kBatchSize = 10000;
  static constexpr int kGapSamples = 8;
  static constexpr int kPulseSamples = 2;
  static constexpr double kPulseAmplitude = 10.0;
  static constexpr double kNoiseAmplitude = 0.5;
};

}  // namespace pulsecore
