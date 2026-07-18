#pragma once

#include <cstdint>
#include <vector>

#include "pulse.pb.h"

namespace pulsecore {

// Minimal PRI-based deinterleaver: groups incoming pulses into candidate
// emitter tracks by matching each pulse's arrival time against every
// existing track's predicted next-pulse time (its last pulse plus its
// running PRI estimate). A pulse that doesn't fit any track within
// pri_tolerance_seconds starts a new one; a track's first two pulses are
// always accepted together since there's no PRI to test against yet.
//
// This is a simplified version of classic sequential-PRI deinterleaving
// -- enough to demonstrate the idea against this repo's single-emitter
// synthetic pulse train (which should always converge to exactly one
// track), not a general multi-emitter solver.
class Deinterleaver {
 public:
  Deinterleaver(double sample_rate_hz, double pri_tolerance_seconds);

  void Process(const pulse::PulseEventBatch& batch, pulse::DeinterleaveSummary* out);

 private:
  struct Track {
    uint32_t track_id;
    uint64_t pulse_count = 0;
    uint64_t last_start_sample = 0;
    double pri_sum_seconds = 0.0;
    uint64_t pri_count = 0;
    double peak_sum = 0.0;
  };

  double sample_rate_hz_;
  double pri_tolerance_seconds_;
  std::vector<Track> tracks_;
  uint32_t next_track_id_ = 1;
};

}  // namespace pulsecore
