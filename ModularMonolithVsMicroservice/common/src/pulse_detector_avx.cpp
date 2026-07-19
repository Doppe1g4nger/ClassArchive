#include "pulse_detector_avx.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>

namespace pulsecore {

PulseDetectorAvx::PulseDetectorAvx(double amplitude_threshold, double sample_rate_hz)
    : threshold_sq_(amplitude_threshold * amplitude_threshold), sample_rate_hz_(sample_rate_hz) {}

void PulseDetectorAvx::Process(const pulse::IQBatch& batch, pulse::PulseEventBatch* out) {
  const int n = batch.i_size();
  const double* i_arr = batch.i().data();
  const double* q_arr = batch.q().data();
  const uint64_t first_index = batch.first_sample_index();
  const __m256d threshold_sq_vec = _mm256_set1_pd(threshold_sq_);

  // On the main branch this function had to gather each sample's
  // fields through RepeatedPtrField's one-pointer-per-message layout,
  // which capped it at parity with the scalar detector -- the
  // pointer-chase, not the arithmetic, was the cost (that story is in
  // the main branch's README). The packed columnar schema is what this
  // kernel was waiting for: i[] and q[] are contiguous doubles, so the
  // magnitude/threshold work is straight unmasked vector loads.
  alignas(32) double mag_sq_lane[4];

  int k = 0;
  for (; k + 4 <= n; k += 4) {
    const __m256d iv = _mm256_loadu_pd(i_arr + k);
    const __m256d qv = _mm256_loadu_pd(q_arr + k);
    // Separate multiply then add -- see header for why this must not
    // become a single FMA instruction.
    const __m256d mag_sq = _mm256_add_pd(_mm256_mul_pd(iv, iv), _mm256_mul_pd(qv, qv));
    const __m256d cmp = _mm256_cmp_pd(mag_sq, threshold_sq_vec, _CMP_GE_OQ);
    const int mask = _mm256_movemask_pd(cmp);

    // Fast path: a full vector of below-threshold samples with no open
    // pulse is the common case at this signal's 20% duty cycle and
    // needs no per-lane work at all.
    if (mask == 0 && !in_pulse_) {
      continue;
    }

    _mm256_store_pd(mag_sq_lane, mag_sq);
    for (int j = 0; j < 4; ++j) {
      const bool above = ((mask >> j) & 1) != 0;
      if (above) {
        const double magnitude = std::sqrt(mag_sq_lane[j]);
        if (!in_pulse_) {
          in_pulse_ = true;
          pulse_start_ = first_index + k + j;
          pulse_peak_ = magnitude;
          pulse_sum_ = magnitude;
          pulse_sample_count_ = 1;
        } else {
          pulse_peak_ = std::max(pulse_peak_, magnitude);
          pulse_sum_ += magnitude;
          ++pulse_sample_count_;
        }
      } else if (in_pulse_) {
        out->add_start_sample(pulse_start_);
        out->add_end_sample(first_index + k + j);
        out->add_peak_amplitude(pulse_peak_);
        out->add_mean_amplitude(pulse_sum_ / static_cast<double>(pulse_sample_count_));
        out->add_duration_seconds(static_cast<double>(pulse_sample_count_) / sample_rate_hz_);
        in_pulse_ = false;
      }
    }
  }

  // Scalar tail for n not a multiple of 4 -- identical logic to
  // pulse_detector.cpp's Process(), so bit-identical to it.
  for (; k < n; ++k) {
    const double i = i_arr[k];
    const double q = q_arr[k];
    const double magnitude_sq = i * i + q * q;
    if (magnitude_sq >= threshold_sq_) {
      const double magnitude = std::sqrt(magnitude_sq);
      if (!in_pulse_) {
        in_pulse_ = true;
        pulse_start_ = first_index + k;
        pulse_peak_ = magnitude;
        pulse_sum_ = magnitude;
        pulse_sample_count_ = 1;
      } else {
        pulse_peak_ = std::max(pulse_peak_, magnitude);
        pulse_sum_ += magnitude;
        ++pulse_sample_count_;
      }
    } else if (in_pulse_) {
      out->add_start_sample(pulse_start_);
      out->add_end_sample(first_index + k);
      out->add_peak_amplitude(pulse_peak_);
      out->add_mean_amplitude(pulse_sum_ / static_cast<double>(pulse_sample_count_));
      out->add_duration_seconds(static_cast<double>(pulse_sample_count_) / sample_rate_hz_);
      in_pulse_ = false;
    }
  }
}

}  // namespace pulsecore
