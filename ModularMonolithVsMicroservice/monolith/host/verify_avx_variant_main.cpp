// verify_avx_variant: standalone correctness check for
// pulse_detector_avx.cpp/jammer_avx.cpp, run separately from the usual
// printed-output diffs this repo uses elsewhere because those only show
// three decimal places -- nowhere near enough precision to catch or
// rule out the floating-point reordering jammer_avx.h's docstring
// documents. This runs the scalar and AVX2 versions of both algorithms
// over the same synthetic signal, batch by batch, and reports the
// actual measured relative error in full precision -- for the jammer's
// reordered power sum as always, and (since this branch went
// -ffast-math) for the detector's per-event doubles too, with event
// counts and sample boundaries still required to match exactly.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "iq_source.h"
#include "jammer.h"
#include "jammer_avx.h"
#include "pulse.pb.h"
#include "pulse_detector.h"
#include "pulse_detector_avx.h"

namespace {

// Theoretical-limits branch: the detector gate is tolerance-based, not
// bit-exact. Under the global -ffast-math the scalar and AVX builds
// may contract/reassociate the amplitude arithmetic differently, so
// "same bits" stopped being a claim either build makes. What still
// must hold exactly: same event count and same integer boundaries
// (start/end samples) -- amplitude wobble of ~1 ulp cannot flip a
// threshold crossing at this signal's ~6x margin, and this check
// proves that, run after run. The double-valued fields get the same
// relative gate the jammer has always used, and the max observed
// error is printed so the "within tolerance" claim stays a number,
// not an assertion.
bool CheckDetector(int num_pulses, double tolerance) {
  constexpr double kSampleRateHz = 10000000.0;
  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);
  pulsecore::PulseDetector scalar_detector(6.0, kSampleRateHz);
  pulsecore::PulseDetectorAvx avx_detector(6.0, kSampleRateHz);

  pulse::IQBatch batch;
  bool ok = true;
  int batches = 0;
  double max_rel_err = 0.0;
  while (source.NextBatch(&batch)) {
    pulse::PulseEventBatch scalar_events;
    pulse::PulseEventBatch avx_events;
    scalar_detector.Process(batch, &scalar_events);
    avx_detector.Process(batch, &avx_events);

    if (scalar_events.start_sample_size() != avx_events.start_sample_size()) {
      std::printf("  batch %d: event count mismatch (%d vs %d events)\n", batches,
                  scalar_events.start_sample_size(), avx_events.start_sample_size());
      ok = false;
      ++batches;
      continue;
    }
    for (int k = 0; k < scalar_events.start_sample_size(); ++k) {
      if (scalar_events.start_sample(k) != avx_events.start_sample(k) ||
          scalar_events.end_sample(k) != avx_events.end_sample(k)) {
        std::printf("  batch %d event %d: boundary mismatch\n", batches, k);
        ok = false;
        continue;
      }
      const double fields[3][2] = {
          {scalar_events.peak_amplitude(k), avx_events.peak_amplitude(k)},
          {scalar_events.mean_amplitude(k), avx_events.mean_amplitude(k)},
          {scalar_events.duration_seconds(k), avx_events.duration_seconds(k)},
      };
      for (const auto& f : fields) {
        const double denom = std::fabs(f[0]) > 0.0 ? std::fabs(f[0]) : 1.0;
        const double rel = std::fabs(f[0] - f[1]) / denom;
        if (rel > max_rel_err) max_rel_err = rel;
        if (rel > tolerance) {
          std::printf("  batch %d event %d: field error %.3e exceeds tolerance\n", batches, k, rel);
          ok = false;
        }
      }
    }
    ++batches;
  }
  std::printf(
      "detector (counts/boundaries exact, values within %.0e, %d batches): %s "
      "(max relative error %.3e)\n",
      tolerance, batches, ok ? "OK" : "MISMATCH", max_rel_err);
  return ok;
}

bool CheckJammer(int num_pulses, double tolerance) {
  constexpr double kSampleRateHz = 10000000.0;
  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);
  pulsecore::JammerDetector scalar_jammer(20.0, 0.5);
  pulsecore::JammerDetectorAvx avx_jammer(20.0, 0.5);

  pulse::IQBatch batch;
  pulse::JamSummary scalar_out;
  pulse::JamSummary avx_out;
  while (source.NextBatch(&batch)) {
    scalar_jammer.Process(batch, &scalar_out);
    avx_jammer.Process(batch, &avx_out);
  }

  const bool exact_fields = scalar_out.batches_total() == avx_out.batches_total() &&
                             scalar_out.batches_flagged() == avx_out.batches_flagged() &&
                             scalar_out.max_duty_cycle() == avx_out.max_duty_cycle();
  const double rel_err =
      std::fabs(scalar_out.max_mean_power() - avx_out.max_mean_power()) / scalar_out.max_mean_power();

  std::printf("jammer batches_total/batches_flagged/max_duty_cycle (exact match required): %s\n",
              exact_fields ? "OK" : "MISMATCH");
  std::printf("jammer max_mean_power relative error: %.3e (tolerance: %.0e)\n", rel_err, tolerance);

  return exact_fields && rel_err < tolerance;
}

}  // namespace

int main(int argc, char** argv) {
  const int num_pulses = argc > 1 ? std::atoi(argv[1]) : 50000;
  constexpr double kTolerance = 1e-9;

  const bool detector_ok = CheckDetector(num_pulses, kTolerance);
  std::printf("\n");
  const bool jammer_ok = CheckJammer(num_pulses, kTolerance);
  std::printf("\n%s\n", (detector_ok && jammer_ok) ? "PASS" : "FAIL");
  return (detector_ok && jammer_ok) ? 0 : 1;
}
