#pragma once

#include <cstdint>
#include <vector>

#include "pulse.pb.h"

namespace pulsecore {

// Small-bin magnitude spectrum estimator with a running accumulator.
// Computes each bin as a direct correlation against a complex tone (a
// single-frequency Goertzel-style DFT term) rather than a full FFT --
// with only a handful of bins that's simpler to read and plenty fast
// for a demo, at the cost of scaling as O(num_bins * batch_size) instead
// of O(batch_size * log(batch_size)).
//
// Like PulseStatsAccumulator, Process() folds one batch into running
// state and reports the summary computed so far, so it can sit directly
// behind the same one-call-per-batch module contract.
class SpectrogramAnalyzer {
 public:
  // Theoretical-limits branch: an analyzer may own just a RANGE of the
  // bins, [bin_begin, bin_end), so two instances on two threads can
  // split the stage's work -- each bin's correlator is fully
  // independent (own phasor, own accumulators, own output slots), so
  // splitting reorders nothing within any bin. bin_hz is still derived
  // from num_bins (the TOTAL), so a range instance computes exactly
  // the same frequencies the full instance would.
  //
  // Output contract: a full-range instance owns the whole summary and
  // rebuilds it every call (the original behavior). A range instance
  // writes ONLY its bins' entries -- the caller must pre-size the
  // summary's arrays to num_bins (so concurrent range instances touch
  // disjoint memory; nothing resizes, nothing races) -- and only the
  // instance owning bin 0 writes the summary's scalar fields.
  SpectrogramAnalyzer(double sample_rate_hz, int num_bins = 8, int bin_begin = -1,
                      int bin_end = -1);

  void Process(const pulse::IQBatch& batch, pulse::SpectrogramSummary* out);

 private:
  double sample_rate_hz_;
  int num_bins_;
  int bin_begin_;
  int bin_end_;
  bool full_range_;
  double bin_hz_;

  std::vector<double> max_magnitude_;
  std::vector<double> sum_magnitude_;
  uint64_t frame_count_ = 0;

  // Cached per-bin offset phasor tables e^{-j*omega*k}, rebuilt only
  // when the batch length changes -- see Process() for the factoring
  // that makes the per-batch work transcendental-free.
  int table_n_ = -1;
  std::vector<double> table_re_;
  std::vector<double> table_im_;
};

}  // namespace pulsecore
