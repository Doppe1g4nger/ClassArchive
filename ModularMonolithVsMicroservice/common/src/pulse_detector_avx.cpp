#include "pulse_detector_avx.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>

namespace pulsecore {

PulseDetectorAvx::PulseDetectorAvx(double amplitude_threshold, double sample_rate_hz)
    : threshold_sq_(amplitude_threshold * amplitude_threshold), sample_rate_hz_(sample_rate_hz) {}

void PulseDetectorAvx::Process(const pulse::IQBatch& batch, pulse::PulseEventBatch* out) {
  const int n = batch.samples_size();
  const __m256d threshold_sq_vec = _mm256_set1_pd(threshold_sq_);

  // Fused, chunk-of-4 version: an earlier version of this function
  // extracted i()/q()/sample_index() for the *whole* batch into
  // heap-allocated arrays first, then ran the vectorized compare as a
  // separate pass, then the state machine as a third pass over those
  // arrays. That measured slower than the plain scalar detector, not
  // faster -- profiling why is what led here. protobuf's repeated
  // IQSample is a repeated *message* field: RepeatedPtrField stores
  // pointers to separately heap-allocated IQSample objects, not a
  // contiguous double[], so reading a sample's fields means chasing a
  // pointer no matter how the arithmetic afterward is done. That
  // pointer-chase -- not the few FLOPs of i*i+q*q -- is this loop's
  // actual cost, so vectorizing only the arithmetic while adding two
  // full extra read/write passes over batch-sized arrays made things
  // slower, not faster: real memory traffic paid for a compute
  // bottleneck that was never there. This version keeps everything in
  // four-element, register-resident scratch instead of batch-sized
  // arrays, so it's back to one pass over the pointers -- matching the
  // scalar version's memory behavior -- with the compare itself still
  // vectorized. See scripts/verify_avx_variant.sh's numbers for whether
  // that was enough to close the gap.
  alignas(32) double i_lane[4];
  alignas(32) double q_lane[4];
  alignas(32) double mag_sq_lane[4];
  uint64_t idx_lane[4];

  int k = 0;
  for (; k + 4 <= n; k += 4) {
    for (int j = 0; j < 4; ++j) {
      const pulse::IQSample& s = batch.samples(k + j);
      i_lane[j] = s.i();
      q_lane[j] = s.q();
      idx_lane[j] = s.sample_index();
    }

    const __m256d iv = _mm256_load_pd(i_lane);
    const __m256d qv = _mm256_load_pd(q_lane);
    // Separate multiply then add -- see header for why this must not
    // become a single FMA instruction.
    const __m256d mag_sq = _mm256_add_pd(_mm256_mul_pd(iv, iv), _mm256_mul_pd(qv, qv));
    _mm256_store_pd(mag_sq_lane, mag_sq);
    const __m256d cmp = _mm256_cmp_pd(mag_sq, threshold_sq_vec, _CMP_GE_OQ);
    const int mask = _mm256_movemask_pd(cmp);

    for (int j = 0; j < 4; ++j) {
      const bool above = ((mask >> j) & 1) != 0;
      if (above) {
        const double magnitude = std::sqrt(mag_sq_lane[j]);
        if (!in_pulse_) {
          in_pulse_ = true;
          pulse_start_ = idx_lane[j];
          pulse_peak_ = magnitude;
          pulse_sum_ = magnitude;
          pulse_sample_count_ = 1;
        } else {
          pulse_peak_ = std::max(pulse_peak_, magnitude);
          pulse_sum_ += magnitude;
          ++pulse_sample_count_;
        }
      } else if (in_pulse_) {
        pulse::PulseEvent* event = out->add_events();
        event->set_start_sample(pulse_start_);
        event->set_end_sample(idx_lane[j]);
        event->set_peak_amplitude(pulse_peak_);
        event->set_mean_amplitude(pulse_sum_ / static_cast<double>(pulse_sample_count_));
        event->set_duration_seconds(static_cast<double>(pulse_sample_count_) / sample_rate_hz_);
        in_pulse_ = false;
      }
    }
  }

  // Scalar tail for n not a multiple of 4 -- identical logic to
  // pulse_detector.cpp's Process(), so bit-identical to it.
  for (; k < n; ++k) {
    const pulse::IQSample& s = batch.samples(k);
    const double i = s.i();
    const double q = s.q();
    const double magnitude_sq = i * i + q * q;
    if (magnitude_sq >= threshold_sq_) {
      const double magnitude = std::sqrt(magnitude_sq);
      if (!in_pulse_) {
        in_pulse_ = true;
        pulse_start_ = s.sample_index();
        pulse_peak_ = magnitude;
        pulse_sum_ = magnitude;
        pulse_sample_count_ = 1;
      } else {
        pulse_peak_ = std::max(pulse_peak_, magnitude);
        pulse_sum_ += magnitude;
        ++pulse_sample_count_;
      }
    } else if (in_pulse_) {
      pulse::PulseEvent* event = out->add_events();
      event->set_start_sample(pulse_start_);
      event->set_end_sample(s.sample_index());
      event->set_peak_amplitude(pulse_peak_);
      event->set_mean_amplitude(pulse_sum_ / static_cast<double>(pulse_sample_count_));
      event->set_duration_seconds(static_cast<double>(pulse_sample_count_) / sample_rate_hz_);
      in_pulse_ = false;
    }
  }
}

}  // namespace pulsecore
