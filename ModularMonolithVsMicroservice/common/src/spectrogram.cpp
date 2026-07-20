#include "spectrogram.h"

#include <algorithm>
#include <cmath>

namespace pulsecore {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

SpectrogramAnalyzer::SpectrogramAnalyzer(double sample_rate_hz, int num_bins, int bin_begin,
                                         int bin_end)
    : sample_rate_hz_(sample_rate_hz),
      num_bins_(num_bins),
      bin_begin_(bin_begin < 0 ? 0 : bin_begin),
      bin_end_(bin_end < 0 ? num_bins : bin_end),
      full_range_(bin_begin_ == 0 && bin_end_ == num_bins),
      bin_hz_(sample_rate_hz / (2.0 * num_bins)),
      max_magnitude_(num_bins, 0.0),
      sum_magnitude_(num_bins, 0.0) {}

void SpectrogramAnalyzer::Process(const pulse::IQBatch& batch, pulse::SpectrogramSummary* out) {
  const int n = batch.i_size();
  if (n > 0) {
    const double* i_arr = batch.i().data();
    const double* q_arr = batch.q().data();

    // Per-bin offset tables e^{-j*omega*k}, built once per batch length
    // (this repo has exactly two: 10,000 and the 8-sample tail) -- the
    // max-optimization branch's port of the numpy variant's phase-table
    // fix. The absolute phase omega*(first+k) factors into a per-batch
    // scalar phasor e^{-j*omega*first} times the cached table, so the
    // per-(bin,sample) work below is pure multiply/add over four
    // contiguous double arrays -- exactly the shape the compiler's
    // auto-vectorizer wants (see CMakeLists.txt for the reduction-math
    // flags on this file, and the tolerance note in the branch docs:
    // reassociated reductions are not bit-identical to the main
    // branch's recursive rotation, they are printed-output-identical
    // and tolerance-verified).
    if (n != table_n_) {
      table_n_ = n;
      table_re_.assign(static_cast<size_t>(num_bins_) * n, 0.0);
      table_im_.assign(static_cast<size_t>(num_bins_) * n, 0.0);
      for (int bin = bin_begin_; bin < bin_end_; ++bin) {
        const double omega = 2.0 * kPi * ((bin + 0.5) * bin_hz_) / sample_rate_hz_;
        double* tr = &table_re_[static_cast<size_t>(bin) * n];
        double* ti = &table_im_[static_cast<size_t>(bin) * n];
        for (int k = 0; k < n; ++k) {
          tr[k] = std::cos(omega * k);
          ti[k] = -std::sin(omega * k);
        }
      }
    }

    const double first = static_cast<double>(batch.first_sample_index());
    for (int bin = bin_begin_; bin < bin_end_; ++bin) {
      const double omega = 2.0 * kPi * ((bin + 0.5) * bin_hz_) / sample_rate_hz_;
      const double phase0 = omega * first;
      const double r0_re = std::cos(phase0);
      const double r0_im = -std::sin(phase0);
      const double* tr = &table_re_[static_cast<size_t>(bin) * n];
      const double* ti = &table_im_[static_cast<size_t>(bin) * n];

      double re = 0.0;
      double im = 0.0;
      for (int k = 0; k < n; ++k) {
        // rot = (r0_re + j*r0_im) * (tr[k] + j*ti[k]); correlate
        // (i + j*q) against it.
        const double rot_re = r0_re * tr[k] - r0_im * ti[k];
        const double rot_im = r0_re * ti[k] + r0_im * tr[k];
        re += i_arr[k] * rot_re - q_arr[k] * rot_im;
        im += i_arr[k] * rot_im + q_arr[k] * rot_re;
      }

      const double magnitude = std::sqrt(re * re + im * im) / static_cast<double>(n);
      max_magnitude_[bin] = std::max(max_magnitude_[bin], magnitude);
      sum_magnitude_[bin] += magnitude;
    }
    ++frame_count_;
  }

  if (full_range_) {
    // Sole owner of the summary: rebuild it whole (original behavior).
    out->Clear();
    out->set_bin_hz(bin_hz_);
    out->set_frame_count(frame_count_);
    for (int bin = 0; bin < num_bins_; ++bin) {
      out->add_max_magnitude(max_magnitude_[bin]);
      out->add_mean_magnitude(
          frame_count_ > 0 ? sum_magnitude_[bin] / static_cast<double>(frame_count_) : 0.0);
    }
    return;
  }

  // Range instance: another instance owns the other bins, possibly on
  // another thread THIS call overlaps with. Touch only this range's
  // pre-sized entries (see the header's output contract -- nothing
  // here may resize or Clear), and let the bin-0 owner write the
  // scalars so every field has exactly one writer.
  if (bin_begin_ == 0) {
    out->set_bin_hz(bin_hz_);
    out->set_frame_count(frame_count_);
  }
  for (int bin = bin_begin_; bin < bin_end_; ++bin) {
    out->set_max_magnitude(bin, max_magnitude_[bin]);
    out->set_mean_magnitude(
        bin, frame_count_ > 0 ? sum_magnitude_[bin] / static_cast<double>(frame_count_) : 0.0);
  }
}

}  // namespace pulsecore
