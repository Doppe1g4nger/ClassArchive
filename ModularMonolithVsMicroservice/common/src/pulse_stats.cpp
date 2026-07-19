#include "pulse_stats.h"

#include <algorithm>

namespace pulsecore {

PulseStatsAccumulator::PulseStatsAccumulator(double sample_rate_hz)
    : sample_rate_hz_(sample_rate_hz) {}

void PulseStatsAccumulator::Add(const pulse::PulseEventBatch& batch) {
  // Columnar events (see pulse.proto): entry k across the parallel
  // arrays is one pulse; iteration walks contiguous doubles.
  const int n = batch.start_sample_size();
  for (int k = 0; k < n; ++k) {
    const double peak = batch.peak_amplitude(k);
    ++count_;
    peak_sum_ += peak;
    duration_sum_ += batch.duration_seconds(k);
    peak_min_ = std::min(peak_min_, peak);
    peak_max_ = std::max(peak_max_, peak);

    if (have_prev_start_) {
      const double gap_samples =
          static_cast<double>(batch.start_sample(k) - prev_start_sample_);
      pri_sum_ += gap_samples / sample_rate_hz_;
      ++pri_count_;
    }
    prev_start_sample_ = batch.start_sample(k);
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
