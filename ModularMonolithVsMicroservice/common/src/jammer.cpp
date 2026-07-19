#include "jammer.h"

#include <algorithm>

namespace pulsecore {

JammerDetector::JammerDetector(double power_threshold, double duty_cycle_threshold)
    : power_threshold_(power_threshold), duty_cycle_threshold_(duty_cycle_threshold) {}

void JammerDetector::Process(const pulse::IQBatch& batch, pulse::JamSummary* out) {
  const int n = batch.i_size();
  if (n > 0) {
    const double* i_arr = batch.i().data();
    const double* q_arr = batch.q().data();
    double power_sum = 0.0;
    uint64_t over_threshold = 0;
    for (int k = 0; k < n; ++k) {
      const double power = i_arr[k] * i_arr[k] + q_arr[k] * q_arr[k];
      power_sum += power;
      if (power >= power_threshold_) ++over_threshold;
    }

    const double mean_power = power_sum / static_cast<double>(n);
    const double duty_cycle =
        static_cast<double>(over_threshold) / static_cast<double>(n);

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
