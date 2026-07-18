#include "pulse_stats.h"

#include <algorithm>

namespace pulsecore {

PulseStatsAccumulator::PulseStatsAccumulator(double sample_rate_hz)
    : sample_rate_hz_(sample_rate_hz) {}

void PulseStatsAccumulator::Add(const pulse::PulseEventBatch& batch) {
  for (const pulse::PulseEvent& e : batch.events()) {
    ++count_;
    peak_sum_ += e.peak_amplitude();
    duration_sum_ += e.duration_seconds();
    peak_min_ = std::min(peak_min_, e.peak_amplitude());
    peak_max_ = std::max(peak_max_, e.peak_amplitude());

    if (have_prev_start_) {
      const double gap_samples =
          static_cast<double>(e.start_sample() - prev_start_sample_);
      pri_sum_ += gap_samples / sample_rate_hz_;
      ++pri_count_;
    }
    prev_start_sample_ = e.start_sample();
    have_prev_start_ = true;
  }
}

pulse::PulseSummary PulseStatsAccumulator::Finalize() const {
  pulse::PulseSummary summary;
  summary.set_pulse_count(count_);
  if (count_ > 0) {
    summary.set_mean_peak_amplitude(peak_sum_ / static_cast<double>(count_));
    summary.set_mean_duration_seconds(duration_sum_ / static_cast<double>(count_));
    summary.set_min_peak_amplitude(peak_min_);
    summary.set_max_peak_amplitude(peak_max_);
  }
  if (pri_count_ > 0) {
    summary.set_mean_pri_seconds(pri_sum_ / static_cast<double>(pri_count_));
  }
  return summary;
}

}  // namespace pulsecore
