#pragma once

#include "pulse.pb.h"

namespace pulsecore {

// Simple magnitude-threshold pulse detector with state carried across
// batches so a pulse can straddle a batch boundary without being split
// into two. This is the one piece of "business logic" in the whole demo;
// it is identical whether it ends up inside a dynamically loaded .so or
// inside a standalone microservice binary.
class PulseDetector {
 public:
  PulseDetector(double amplitude_threshold, double sample_rate_hz);

  void Process(const pulse::IQBatch& batch, pulse::PulseEventBatch* out);

 private:
  double threshold_;
  double sample_rate_hz_;

  bool in_pulse_ = false;
  uint64_t pulse_start_ = 0;
  double pulse_peak_ = 0.0;
  double pulse_sum_ = 0.0;
  uint64_t pulse_sample_count_ = 0;
};

}  // namespace pulsecore
