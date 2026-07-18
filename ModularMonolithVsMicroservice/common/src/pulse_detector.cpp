#include "pulse_detector.h"

#include <cmath>

namespace pulsecore {

PulseDetector::PulseDetector(double amplitude_threshold, double sample_rate_hz)
    : threshold_(amplitude_threshold), sample_rate_hz_(sample_rate_hz) {}

void PulseDetector::Process(const pulse::IQBatch& batch, pulse::PulseEventBatch* out) {
  for (const pulse::IQSample& s : batch.samples()) {
    const double magnitude = std::sqrt(s.i() * s.i() + s.q() * s.q());
    const bool above = magnitude >= threshold_;

    if (above && !in_pulse_) {
      in_pulse_ = true;
      pulse_start_ = s.sample_index();
      pulse_peak_ = magnitude;
      pulse_sum_ = magnitude;
      pulse_sample_count_ = 1;
    } else if (above && in_pulse_) {
      pulse_peak_ = std::max(pulse_peak_, magnitude);
      pulse_sum_ += magnitude;
      ++pulse_sample_count_;
    } else if (!above && in_pulse_) {
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
