#pragma once

#include <cstdint>

#include "pulse.pb.h"

namespace pulsecore {

// Flags a batch as "jammed" when a large fraction of its samples sit
// above a power threshold -- a crude stand-in for wideband/barrage
// jamming detection. Real jamming rarely looks like the detector's
// narrow rectangular pulses; it looks like elevated energy sustained
// across most of a batch, which is what duty_cycle_threshold measures
// against. The defaults are set well above this repo's synthetic pulse
// train's normal duty cycle (20%, since each 10-sample period is 2
// samples of pulse -- see iq_source.h) specifically so ordinary pulsed
// traffic doesn't get misreported as jamming.
class JammerDetector {
 public:
  JammerDetector(double power_threshold, double duty_cycle_threshold);

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
