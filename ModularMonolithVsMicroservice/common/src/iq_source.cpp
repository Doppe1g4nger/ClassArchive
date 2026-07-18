#include "iq_source.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pulsecore {

SyntheticIQSource::SyntheticIQSource(double sample_rate_hz, int num_pulses, uint32_t seed)
    : sample_rate_hz_(sample_rate_hz),
      num_pulses_(num_pulses),
      total_samples_(static_cast<uint64_t>(num_pulses) * (kGapSamples + kPulseSamples) +
                      kGapSamples),
      rng_state_(seed == 0 ? 1 : seed) {}

double SyntheticIQSource::NextNoise() {
  // xorshift32: tiny, deterministic, good enough for demo noise.
  rng_state_ ^= rng_state_ << 13;
  rng_state_ ^= rng_state_ >> 17;
  rng_state_ ^= rng_state_ << 5;
  const double unit = static_cast<double>(rng_state_) / static_cast<double>(UINT32_MAX);
  return (unit - 0.5) * 2.0 * kNoiseAmplitude;
}

bool SyntheticIQSource::NextBatch(pulse::IQBatch* batch) {
  if (sample_cursor_ >= total_samples_) {
    return false;
  }

  batch->Clear();
  batch->set_sample_rate_hz(sample_rate_hz_);

  const uint64_t period = kGapSamples + kPulseSamples;
  const int count = static_cast<int>(
      std::min<uint64_t>(kBatchSize, total_samples_ - sample_cursor_));

  for (int i = 0; i < count; ++i) {
    const uint64_t idx = sample_cursor_ + i;
    const uint64_t phase = idx % period;
    const bool in_pulse = phase >= kGapSamples;

    const double amplitude = in_pulse ? kPulseAmplitude : 0.0;
    // Split amplitude evenly across I/Q so magnitude sqrt(i^2+q^2) == amplitude.
    const double component = amplitude / std::sqrt(2.0);

    pulse::IQSample* s = batch->add_samples();
    s->set_sample_index(idx);
    s->set_i(component + NextNoise());
    s->set_q(component + NextNoise());
  }

  sample_cursor_ += count;
  return true;
}

}  // namespace pulsecore
