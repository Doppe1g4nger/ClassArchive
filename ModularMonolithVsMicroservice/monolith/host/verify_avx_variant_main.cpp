// verify_avx_variant: standalone correctness check for
// pulse_detector_avx.cpp/jammer_avx.cpp, run separately from the usual
// printed-output diffs this repo uses elsewhere because those only show
// three decimal places -- nowhere near enough precision to catch or
// rule out the floating-point reordering jammer_avx.h's docstring
// documents. This runs the scalar and AVX2 versions of both algorithms
// over the same synthetic signal, batch by batch, and reports the
// actual measured relative error in full precision, plus an exact
// (bit-for-bit) equality check for every field the detector emits.

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

bool CheckDetector(int num_pulses) {
  constexpr double kSampleRateHz = 10000000.0;
  pulsecore::SyntheticIQSource source(kSampleRateHz, num_pulses);
  pulsecore::PulseDetector scalar_detector(6.0, kSampleRateHz);
  pulsecore::PulseDetectorAvx avx_detector(6.0, kSampleRateHz);

  pulse::IQBatch batch;
  bool exact = true;
  int batches = 0;
  while (source.NextBatch(&batch)) {
    pulse::PulseEventBatch scalar_events;
    pulse::PulseEventBatch avx_events;
    scalar_detector.Process(batch, &scalar_events);
    avx_detector.Process(batch, &avx_events);

    // Columnar events: serialized equality covers every field of every
    // event in order (packed scalar fields serialize deterministically).
    if (scalar_events.SerializeAsString() != avx_events.SerializeAsString()) {
      std::printf("  batch %d: event batch mismatch (%d vs %d events)\n", batches,
                  scalar_events.start_sample_size(), avx_events.start_sample_size());
      exact = false;
    }
    ++batches;
  }
  std::printf("detector (exact match required, %d batches): %s\n", batches,
              exact ? "OK" : "MISMATCH");
  return exact;
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

  const bool detector_ok = CheckDetector(num_pulses);
  std::printf("\n");
  const bool jammer_ok = CheckJammer(num_pulses, kTolerance);
  std::printf("\n%s\n", (detector_ok && jammer_ok) ? "PASS" : "FAIL");
  return (detector_ok && jammer_ok) ? 0 : 1;
}
