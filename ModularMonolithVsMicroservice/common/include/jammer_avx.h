#pragma once

#include <cstdint>

#include "pulse.pb.h"

namespace pulsecore {

// AVX2 port of JammerDetector (jammer.h). Unlike PulseDetectorAvx, this
// one is *not* bit-identical to the scalar version, and deliberately so
// -- see pulse_detector_avx.h for the case where reordering doesn't
// happen; here it does. power_sum is a reduction over all n samples in
// the batch: the vectorized version accumulates four running partial
// sums (one per SIMD lane) and combines them at the end, which changes
// the order additions happen in relative to the scalar loop's single
// running total. Floating-point addition isn't associative, so a
// different order can change the last few bits of the result even
// though the same additions, of the same values, are being performed.
// This is the same phenomenon numpy_variant/kernels.py's module
// docstring documents for numpy's pairwise summation -- verified to
// stay within a relative error of about 1e-13 at this repo's scale (see
// scripts/verify_avx_variant.sh), not just assumed safe.
class JammerDetectorAvx {
 public:
  JammerDetectorAvx(double power_threshold, double duty_cycle_threshold);

  void Process(const pulse::IQBatch& batch, pulse::JamSummary* out);

 private:
  double power_threshold_;
  double duty_cycle_threshold_;

  uint64_t batches_total_ = 0;
  uint64_t batches_flagged_ = 0;
  double max_duty_cycle_ = 0.0;
  double max_mean_power_ = 0.0;
};

}  // namespace pulsecore
