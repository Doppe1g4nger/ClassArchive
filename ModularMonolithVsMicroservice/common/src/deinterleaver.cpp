#include "deinterleaver.h"

#include <cmath>

namespace pulsecore {

Deinterleaver::Deinterleaver(double sample_rate_hz, double pri_tolerance_seconds)
    : sample_rate_hz_(sample_rate_hz), pri_tolerance_seconds_(pri_tolerance_seconds) {}

void Deinterleaver::Process(const pulse::PulseEventBatch& batch, pulse::DeinterleaveSummary* out) {
  // Columnar events (see pulse.proto): entry e across the parallel
  // arrays is one pulse.
  const int n_events = batch.start_sample_size();
  for (int e = 0; e < n_events; ++e) {
    const uint64_t start_sample = batch.start_sample(e);
    const double pulse_time = static_cast<double>(start_sample) / sample_rate_hz_;

    // Prefer the closest track whose predicted next-pulse time falls
    // within tolerance; fall back to a track that only has one pulse so
    // far (no PRI to test against yet) if no established track matches.
    Track* best_match = nullptr;
    double best_diff = pri_tolerance_seconds_;
    Track* seed_match = nullptr;

    for (Track& track : tracks_) {
      if (track.pri_count > 0) {
        const double last_time = static_cast<double>(track.last_start_sample) / sample_rate_hz_;
        const double predicted =
            last_time + (track.pri_sum_seconds / static_cast<double>(track.pri_count));
        const double diff = std::abs(predicted - pulse_time);
        if (diff <= best_diff) {
          best_diff = diff;
          best_match = &track;
        }
      } else if (track.pulse_count == 1 && seed_match == nullptr) {
        seed_match = &track;
      }
    }

    Track* target = best_match != nullptr ? best_match : seed_match;
    if (target == nullptr) {
      tracks_.push_back(Track{next_track_id_++});
      target = &tracks_.back();
    }

    if (target->pulse_count > 0) {
      const double last_time = static_cast<double>(target->last_start_sample) / sample_rate_hz_;
      target->pri_sum_seconds += (pulse_time - last_time);
      ++target->pri_count;
    }
    target->last_start_sample = start_sample;
    target->peak_sum += batch.peak_amplitude(e);
    ++target->pulse_count;
  }

  out->Clear();
  for (const Track& track : tracks_) {
    pulse::EmitterTrack* t = out->add_tracks();
    t->set_track_id(track.track_id);
    t->set_pulse_count(track.pulse_count);
    t->set_estimated_pri_seconds(
        track.pri_count > 0 ? track.pri_sum_seconds / static_cast<double>(track.pri_count) : 0.0);
    t->set_mean_peak_amplitude(
        track.pulse_count > 0 ? track.peak_sum / static_cast<double>(track.pulse_count) : 0.0);
  }
}

}  // namespace pulsecore
