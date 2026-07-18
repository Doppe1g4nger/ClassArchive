#include "jammer_avx.h"

#include <immintrin.h>

#include <algorithm>

namespace pulsecore {

namespace {

double HorizontalSum(__m256d v) {
  const __m128d lo = _mm256_castpd256_pd128(v);
  const __m128d hi = _mm256_extractf128_pd(v, 1);
  const __m128d sum128 = _mm_add_pd(lo, hi);
  const __m128d hi64 = _mm_unpackhi_pd(sum128, sum128);
  return _mm_cvtsd_f64(_mm_add_sd(sum128, hi64));
}

}  // namespace

JammerDetectorAvx::JammerDetectorAvx(double power_threshold, double duty_cycle_threshold)
    : power_threshold_(power_threshold), duty_cycle_threshold_(duty_cycle_threshold) {}

void JammerDetectorAvx::Process(const pulse::IQBatch& batch, pulse::JamSummary* out) {
  const int n = batch.samples_size();
  if (n > 0) {
    const __m256d threshold_vec = _mm256_set1_pd(power_threshold_);
    // Four independent running sums (one per lane) -- explicitly *not*
    // the same accumulation order as the scalar version's single
    // running total. See the header for why that's an accepted,
    // measured tradeoff here, unlike the detector.
    __m256d power_acc = _mm256_setzero_pd();
    uint64_t over_threshold = 0;

    // Four-element, register-resident scratch instead of a batch-sized
    // buffer -- see pulse_detector_avx.cpp for why: extracting the
    // *whole* batch into heap arrays first (an earlier version of this
    // function did exactly that) adds real memory traffic to pay for a
    // compute bottleneck that was never there, since protobuf's
    // pointer-chase through each IQSample -- not the arithmetic -- is
    // what this loop actually costs.
    alignas(32) double i_lane[4];
    alignas(32) double q_lane[4];

    int k = 0;
    for (; k + 4 <= n; k += 4) {
      for (int j = 0; j < 4; ++j) {
        const pulse::IQSample& s = batch.samples(k + j);
        i_lane[j] = s.i();
        q_lane[j] = s.q();
      }
      const __m256d iv = _mm256_load_pd(i_lane);
      const __m256d qv = _mm256_load_pd(q_lane);
      const __m256d power = _mm256_fmadd_pd(iv, iv, _mm256_mul_pd(qv, qv));
      power_acc = _mm256_add_pd(power_acc, power);

      const __m256d cmp = _mm256_cmp_pd(power, threshold_vec, _CMP_GE_OQ);
      over_threshold += static_cast<uint64_t>(__builtin_popcount(
          static_cast<unsigned>(_mm256_movemask_pd(cmp))));
    }

    double power_sum = HorizontalSum(power_acc);
    for (; k < n; ++k) {
      const pulse::IQSample& s = batch.samples(k);
      const double i = s.i();
      const double q = s.q();
      const double power = i * i + q * q;
      power_sum += power;
      if (power >= power_threshold_) ++over_threshold;
    }

    const double mean_power = power_sum / static_cast<double>(n);
    const double duty_cycle = static_cast<double>(over_threshold) / static_cast<double>(n);

    ++batches_total_;
    if (duty_cycle >= duty_cycle_threshold_) ++batches_flagged_;
    max_duty_cycle_ = std::max(max_duty_cycle_, duty_cycle);
    max_mean_power_ = std::max(max_mean_power_, mean_power);
  }

  out->Clear();
  out->set_batches_total(batches_total_);
  out->set_batches_flagged(batches_flagged_);
  out->set_max_duty_cycle(max_duty_cycle_);
  out->set_max_mean_power(max_mean_power_);
}

}  // namespace pulsecore
