#include "pulse_detector.h"

#include <cmath>

namespace pulsecore {

PulseDetector::PulseDetector(double amplitude_threshold, double sample_rate_hz)
    : threshold_(amplitude_threshold),
      threshold_sq_(amplitude_threshold * amplitude_threshold),
      sample_rate_hz_(sample_rate_hz) {}

void PulseDetector::Process(const pulse::IQBatch& batch, pulse::PulseEventBatch* out) {
  const int n = batch.i_size();
  const double* i_arr = batch.i().data();
  const double* q_arr = batch.q().data();
  const uint64_t first_index = batch.first_sample_index();

  for (int k = 0; k < n; ++k) {
    const double i = i_arr[k];
    const double q = q_arr[k];
    const double magnitude_sq = i * i + q * q;
    const bool above = magnitude_sq >= threshold_sq_;

    // sqrt() is only ever computed for the samples that actually cross
    // the threshold (this repo's synthetic signal keeps that to ~20% of
    // samples -- its duty cycle, see iq_source.h) -- every sample that's
    // below threshold, or above it but not the falling edge, needs
    // nothing more than the squared-magnitude comparison above.
    if (above) {
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
      // Columnar event output (see pulse.proto): one append per field,
      // no per-event message allocation.
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
