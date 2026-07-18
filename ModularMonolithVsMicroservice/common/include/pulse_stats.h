#pragma once

#include <cstdint>
#include <limits>

#include "pulse.pb.h"

namespace pulsecore {

// Accumulates PulseEvent batches into running summary statistics. Like
// PulseDetector, this class carries no knowledge of whether it is sitting
// behind a dlopen() boundary or on the far end of a socket -- it only
// speaks protobuf types in and out.
class PulseStatsAccumulator {
 public:
  // sample_rate_hz must match the rate the upstream detector used, since
  // PulseEvent only carries sample indices; converting the gap between
  // consecutive pulse starts into a PRI in seconds needs that rate.
  explicit PulseStatsAccumulator(double sample_rate_hz);

  void Add(const pulse::PulseEventBatch& batch);

  pulse::PulseSummary Finalize() const;

 private:
  double sample_rate_hz_;

  uint64_t count_ = 0;
  double peak_sum_ = 0.0;
  double duration_sum_ = 0.0;
  double peak_min_ = std::numeric_limits<double>::max();
  double peak_max_ = std::numeric_limits<double>::lowest();

  double pri_sum_ = 0.0;
  uint64_t pri_count_ = 0;
  bool have_prev_start_ = false;
  uint64_t prev_start_sample_ = 0;
};

}  // namespace pulsecore
