#pragma once

#include <cstdint>

#include "pulse.pb.h"

namespace pulsecore {

// AVX2 port of PulseDetector (pulse_detector.h) -- built to answer a
// direct follow-up from the correctness-audit pass: "optimize further
// with AVX instructions?" Same algorithm, same output, down to the bit:
// the magnitude-squared/threshold decision for four samples is computed
// at once with 256-bit vector instructions instead of one at a time,
// then the same sequential state machine (pulse start/peak/sum/count,
// carried across batches) runs over the precomputed per-sample results.
//
// Why this stays bit-identical where jammer_avx.h and a hypothetical
// AVX spectrogram would not: the vectorized part here is a pure
// per-element computation (i*i+q*q, then compare), not a reduction.
// IEEE-754 guarantees a multiply or add produces the same result
// regardless of what other SIMD lanes are doing at the same time --
// there's no cross-element combination whose *order* could change,
// unlike a sum. The two vector ops used (separate multiply, then add)
// are also deliberately not fused into one FMA instruction (see
// CMakeLists.txt's -ffp-contract=off for this target) -- an FMA rounds
// once instead of twice and would produce a different last bit than the
// scalar version's separate multiply-then-add, for every sample, not
// just some.
class PulseDetectorAvx {
 public:
  PulseDetectorAvx(double amplitude_threshold, double sample_rate_hz);

  void Process(const pulse::IQBatch& batch, pulse::PulseEventBatch* out);

 private:
  double threshold_sq_;
  double sample_rate_hz_;

  bool in_pulse_ = false;
  uint64_t pulse_start_ = 0;
  double pulse_peak_ = 0.0;
  double pulse_sum_ = 0.0;
  uint64_t pulse_sample_count_ = 0;
};

}  // namespace pulsecore
