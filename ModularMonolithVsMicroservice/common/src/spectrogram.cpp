#include "spectrogram.h"

#include <algorithm>
#include <cmath>

namespace pulsecore {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

SpectrogramAnalyzer::SpectrogramAnalyzer(double sample_rate_hz, int num_bins)
    : sample_rate_hz_(sample_rate_hz),
      num_bins_(num_bins),
      bin_hz_(sample_rate_hz / (2.0 * num_bins)),
      max_magnitude_(num_bins, 0.0),
      sum_magnitude_(num_bins, 0.0) {}

void SpectrogramAnalyzer::Process(const pulse::IQBatch& batch, pulse::SpectrogramSummary* out) {
  if (!batch.samples().empty()) {
    for (int bin = 0; bin < num_bins_; ++bin) {
      // Bin center frequencies are spaced evenly across [0, Nyquist/2);
      // exact placement doesn't matter for this demo, only that it's
      // deterministic and consistent between the two architectures.
      const double freq_hz = (bin + 0.5) * bin_hz_;
      const double omega = 2.0 * kPi * freq_hz / sample_rate_hz_;

      // Rotate a unit phasor by -omega per sample instead of calling
      // cos()/sin() for every one: seed it once at this batch's first
      // sample_index, then advance it with one fixed complex multiply
      // per sample. At 10,000 samples/bin/batch that's the difference
      // between ~4 transcendental calls and ~20,000 per batch -- with 8
      // bins and hundreds of batches, calling cos()/sin() per sample
      // dominated the whole pipeline's runtime. Floating-point drift in
      // the rotator's magnitude over one batch (a few thousand
      // multiplies) is far below anything visible in the output; this
      // wouldn't be safe to run for millions of samples without
      // periodic renormalization, but each batch reseeds from scratch.
      const double start_phase = omega * static_cast<double>(batch.samples(0).sample_index());
      double rot_re = std::cos(start_phase);
      double rot_im = -std::sin(start_phase);  // rot == e^{-j*omega*n}
      const double step_re = std::cos(omega);
      const double step_im = -std::sin(omega);  // step == e^{-j*omega}

      double re = 0.0;
      double im = 0.0;
      for (const pulse::IQSample& s : batch.samples()) {
        // Correlate the complex sample (i + j*q) against the rotator.
        re += s.i() * rot_re - s.q() * rot_im;
        im += s.i() * rot_im + s.q() * rot_re;

        const double next_re = rot_re * step_re - rot_im * step_im;
        const double next_im = rot_re * step_im + rot_im * step_re;
        rot_re = next_re;
        rot_im = next_im;
      }

      const double magnitude = std::sqrt(re * re + im * im) / static_cast<double>(batch.samples_size());
      max_magnitude_[bin] = std::max(max_magnitude_[bin], magnitude);
      sum_magnitude_[bin] += magnitude;
    }
    ++frame_count_;
  }

  out->Clear();
  out->set_bin_hz(bin_hz_);
  out->set_frame_count(frame_count_);
  for (int bin = 0; bin < num_bins_; ++bin) {
    out->add_max_magnitude(max_magnitude_[bin]);
    out->add_mean_magnitude(frame_count_ > 0 ? sum_magnitude_[bin] / static_cast<double>(frame_count_)
                                              : 0.0);
  }
}

}  // namespace pulsecore
