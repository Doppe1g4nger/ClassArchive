// Unit tests for pulsecore (the shared business logic) and netutil (the
// shared-memory ring transport), plus -- when the toolchain built them -- the AVX2
// variant classes. Deliberately framework-free: a tiny CHECK macro and a
// main() that counts failures keeps the repo's dependency footprint at
// zero, which matters more for a teaching repo than gtest's ergonomics
// would. Run via `ctest` (see CMakeLists.txt) or scripts/run_tests.sh.
//
// These complement, not replace, the repo's two other correctness
// layers: the cross-build output diffs documented in README.md's
// "Correctness" section (which prove all eight builds agree end to end)
// and the dedicated verify tools for the AVX2/numpy variants (which
// measure floating-point tolerances the printed output can't expose).
// What unit tests add is coverage for cases the synthetic signal never
// produces at this repo's constants -- a pulse straddling a batch
// boundary, multiple emitters, empty batches -- and pinned golden
// values that fail loudly if either language's generator ever drifts.

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "deinterleaver.h"
#include "framing.h"
#include "iq_source.h"
#include "jammer.h"
#include "pulse.pb.h"
#include "pulse_detector.h"
#include "pulse_stats.h"
#include "spectrogram.h"

#ifdef HAVE_AVX_VARIANT
#include "jammer_avx.h"
#include "pulse_detector_avx.h"
#endif

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      ++failures;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                   \
  } while (0)

#define CHECK_EQ_D(a, b)                                                \
  do {                                                                  \
    const double va = (a);                                              \
    const double vb = (b);                                              \
    if (va != vb) {                                                     \
      ++failures;                                                       \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s  (%.17g vs %.17g)\n",  \
                   __FILE__, __LINE__, #a, #b, va, vb);                 \
    }                                                                   \
  } while (0)

constexpr double kSampleRateHz = 10000000.0;

// Builds an IQBatch where sample k has i=amplitudes[k], q=0 -- so each
// sample's magnitude is exactly amplitudes[k] with no floating-point
// surprises, which keeps expected detector outputs computable by hand.
pulse::IQBatch MakeBatch(const std::vector<double>& amplitudes, uint64_t start_index) {
  pulse::IQBatch batch;
  batch.set_sample_rate_hz(kSampleRateHz);
  batch.set_first_sample_index(start_index);
  for (double amp : amplitudes) {
    batch.add_i(amp);
    batch.add_q(0.0);
  }
  return batch;
}

// ---------------------------------------------------------------------------
// SyntheticIQSource
// ---------------------------------------------------------------------------

void TestIQSourceGoldenValues() {
  // Golden values pinned from the current generator. The Python test
  // suite (python/tests/test_pulsecore.py) asserts these exact same
  // doubles, so a change that silently breaks the two languages'
  // bit-for-bit generator equality fails both suites' goldens rather
  // than passing one and failing a cross-language diff much later.
  pulsecore::SyntheticIQSource source(kSampleRateHz, /*num_pulses=*/10);
  pulse::IQBatch batch;
  CHECK(source.NextBatch(&batch));
  // 10 pulses * 10-sample period + 8 leading gap samples = 108.
  CHECK(batch.i_size() == 108);
  CHECK(batch.q_size() == 108);
  CHECK_EQ_D(batch.sample_rate_hz(), kSampleRateHz);

  CHECK(batch.first_sample_index() == 0);
  CHECK_EQ_D(batch.i(0), -0.4973561074578567);
  CHECK_EQ_D(batch.q(0), 0.16031197753276494);
  CHECK_EQ_D(batch.i(7), -0.044113119725164296);
  CHECK_EQ_D(batch.q(7), -0.2474199623678392);
  // Sample 8 is the first in-pulse sample: 10/sqrt(2) + noise.
  CHECK_EQ_D(batch.i(8), 6.697328498560646);
  CHECK_EQ_D(batch.q(8), 6.5812648301410235);
  CHECK_EQ_D(batch.i(107), -0.31797348238014933);
  CHECK_EQ_D(batch.q(107), 0.484969302077072);

  // One 108-sample signal fits in a single batch.
  CHECK(!source.NextBatch(&batch));
}

void TestIQSourceDeterminismAndBatching() {
  pulsecore::SyntheticIQSource a(kSampleRateHz, 50000);
  pulsecore::SyntheticIQSource b(kSampleRateHz, 50000);
  pulse::IQBatch ba;
  pulse::IQBatch bb;
  int batches = 0;
  uint64_t samples = 0;
  while (a.NextBatch(&ba)) {
    CHECK(b.NextBatch(&bb));
    CHECK(ba.SerializeAsString() == bb.SerializeAsString());
    samples += ba.i_size();
    ++batches;
  }
  CHECK(!b.NextBatch(&bb));
  // 50,000 pulses * 10-sample period + 8 = 500,008 samples in 51
  // batches (50 full 10,000-sample batches + one 8-sample tail).
  CHECK(batches == 51);
  CHECK(samples == 500008);
}

// ---------------------------------------------------------------------------
// PulseDetector
// ---------------------------------------------------------------------------

void TestDetectorBasicPulse() {
  pulsecore::PulseDetector detector(/*amplitude_threshold=*/2.0, kSampleRateHz);
  // Below, below, 3-sample pulse (peaks 5,7,6), below, below.
  pulse::IQBatch batch = MakeBatch({0.5, 0.5, 5.0, 7.0, 6.0, 0.5, 0.5}, 100);
  pulse::PulseEventBatch out;
  detector.Process(batch, &out);

  CHECK(out.start_sample_size() == 1);
  CHECK(out.start_sample(0) == 102);
  CHECK(out.end_sample(0) == 105);  // first below-threshold sample
  CHECK_EQ_D(out.peak_amplitude(0), 7.0);
  CHECK_EQ_D(out.mean_amplitude(0), 6.0);
  CHECK_EQ_D(out.duration_seconds(0), 3.0 / kSampleRateHz);
}

void TestDetectorThresholdIsInclusive() {
  pulsecore::PulseDetector detector(2.0, kSampleRateHz);
  // A sample at exactly the threshold counts as in-pulse (>=, not >).
  pulse::IQBatch batch = MakeBatch({0.0, 2.0, 0.0}, 0);
  pulse::PulseEventBatch out;
  detector.Process(batch, &out);
  CHECK(out.start_sample_size() == 1);
  CHECK_EQ_D(out.peak_amplitude(0), 2.0);
}

void TestDetectorStraddlesBatchBoundary() {
  // The synthetic signal never produces this at this repo's constants
  // (batch size is an exact multiple of the pulse period), but the
  // detector explicitly supports it -- state carries across Process()
  // calls. Split one 4-sample pulse across two batches.
  pulsecore::PulseDetector detector(2.0, kSampleRateHz);
  pulse::IQBatch first = MakeBatch({0.5, 5.0, 6.0}, 0);
  pulse::IQBatch second = MakeBatch({7.0, 4.0, 0.5, 0.5}, 3);
  pulse::PulseEventBatch out;

  detector.Process(first, &out);
  CHECK(out.start_sample_size() == 0);  // pulse still open at batch end

  detector.Process(second, &out);
  CHECK(out.start_sample_size() == 1);
  CHECK(out.start_sample(0) == 1);
  CHECK(out.end_sample(0) == 5);
  CHECK_EQ_D(out.peak_amplitude(0), 7.0);
  CHECK_EQ_D(out.mean_amplitude(0), (5.0 + 6.0 + 7.0 + 4.0) / 4.0);
  CHECK_EQ_D(out.duration_seconds(0), 4.0 / kSampleRateHz);
}

void TestDetectorNoPulses() {
  pulsecore::PulseDetector detector(2.0, kSampleRateHz);
  pulse::IQBatch batch = MakeBatch({0.5, 1.0, 1.9, 0.1}, 0);
  pulse::PulseEventBatch out;
  detector.Process(batch, &out);
  CHECK(out.start_sample_size() == 0);
}

// ---------------------------------------------------------------------------
// PulseStatsAccumulator
// ---------------------------------------------------------------------------

void AddEvent(pulse::PulseEventBatch* batch, uint64_t start, double peak, double duration_s) {
  batch->add_start_sample(start);
  batch->add_end_sample(start + 1);
  batch->add_peak_amplitude(peak);
  batch->add_mean_amplitude(peak);
  batch->add_duration_seconds(duration_s);
}

void TestStatsAccumulator() {
  pulsecore::PulseStatsAccumulator acc(kSampleRateHz);

  pulse::PulseEventBatch batch1;
  AddEvent(&batch1, 0, 5.0, 2e-7);
  AddEvent(&batch1, 10, 7.0, 2e-7);
  acc.Add(batch1);

  // PRI state must carry across Add() calls: the gap from sample 10 to
  // sample 30 spans this batch boundary.
  pulse::PulseEventBatch batch2;
  AddEvent(&batch2, 30, 3.0, 4e-7);
  acc.Add(batch2);

  pulse::PulseSummary s = acc.Finalize();
  CHECK(s.pulse_count() == 3);
  CHECK_EQ_D(s.mean_peak_amplitude(), (5.0 + 7.0 + 3.0) / 3.0);
  CHECK_EQ_D(s.min_peak_amplitude(), 3.0);
  CHECK_EQ_D(s.max_peak_amplitude(), 7.0);
  CHECK_EQ_D(s.mean_duration_seconds(), (2e-7 + 2e-7 + 4e-7) / 3.0);
  // PRIs: 10 samples then 20 samples -> mean 15 samples.
  CHECK_EQ_D(s.mean_pri_seconds(), ((10.0 / kSampleRateHz) + (20.0 / kSampleRateHz)) / 2.0);
}

void TestStatsEmpty() {
  pulsecore::PulseStatsAccumulator acc(kSampleRateHz);
  pulse::PulseSummary s = acc.Finalize();
  CHECK(s.pulse_count() == 0);
  CHECK_EQ_D(s.mean_peak_amplitude(), 0.0);
  CHECK_EQ_D(s.mean_pri_seconds(), 0.0);
}

// ---------------------------------------------------------------------------
// JammerDetector
// ---------------------------------------------------------------------------

void TestJammerDutyCycle() {
  // power_threshold 2.0, duty_cycle_threshold 0.5. Batch 1: 3 of 4
  // samples over power threshold (i^2+q^2 >= 2.0) -> duty 0.75,
  // flagged. Batch 2: 1 of 4 -> duty 0.25, not flagged.
  pulsecore::JammerDetector jammer(2.0, 0.5);
  pulse::JamSummary out;

  pulse::IQBatch hot = MakeBatch({2.0, 3.0, 1.5, 0.1}, 0);  // powers 4, 9, 2.25, 0.01
  jammer.Process(hot, &out);
  CHECK(out.batches_total() == 1);
  CHECK(out.batches_flagged() == 1);
  CHECK_EQ_D(out.max_duty_cycle(), 0.75);
  CHECK_EQ_D(out.max_mean_power(), (4.0 + 9.0 + 2.25 + 0.01) / 4.0);

  pulse::IQBatch cool = MakeBatch({2.0, 0.1, 0.1, 0.1}, 4);
  jammer.Process(cool, &out);
  CHECK(out.batches_total() == 2);
  CHECK(out.batches_flagged() == 1);
  // Maxima must persist from the hotter batch.
  CHECK_EQ_D(out.max_duty_cycle(), 0.75);
  CHECK_EQ_D(out.max_mean_power(), (4.0 + 9.0 + 2.25 + 0.01) / 4.0);
}

// ---------------------------------------------------------------------------
// Deinterleaver
// ---------------------------------------------------------------------------

void TestDeinterleaverSingleEmitter() {
  pulsecore::Deinterleaver deint(kSampleRateHz, /*pri_tolerance_seconds=*/1e-7);
  pulse::PulseEventBatch events;
  // Steady 10-sample PRI.
  for (uint64_t k = 0; k < 5; ++k) {
    AddEvent(&events, 10 * k, 5.0, 2e-7);
  }
  pulse::DeinterleaveSummary out;
  deint.Process(events, &out);
  CHECK(out.tracks_size() == 1);
  CHECK(out.tracks(0).pulse_count() == 5);
  CHECK_EQ_D(out.tracks(0).estimated_pri_seconds(), 10.0 / kSampleRateHz);
  CHECK_EQ_D(out.tracks(0).mean_peak_amplitude(), 5.0);
}

void TestDeinterleaverTwoEmitters() {
  // Two pulse trains, PRIs 10 and 23 samples. Note the construction:
  // emitter A gets its first *two* pulses in before emitter B appears.
  // That's deliberate, and it documents a real property of this
  // simplified sequential-PRI algorithm rather than dodging one: a
  // track's first two pulses are accepted unconditionally (there's no
  // PRI to test against yet), so two emitters whose very first pulses
  // interleave from a cold start get their seeds merged into one
  // track. Once a track has an established PRI, a second emitter's
  // off-cadence pulses fail the tolerance test and correctly seed
  // their own track -- which is what this verifies. (Writing this test
  // with fully interleaved cold-start trains fails, and that failure
  // is the algorithm's documented limitation, not a regression.)
  pulsecore::Deinterleaver deint(kSampleRateHz, 1e-7);
  pulse::PulseEventBatch events;
  std::vector<uint64_t> starts;
  for (uint64_t k = 0; k < 9; ++k) starts.push_back(10 * k);       // A: 0..80, PRI 10
  for (uint64_t k = 0; k < 4; ++k) starts.push_back(13 + 23 * k);  // B: 13,36,59,82
  std::sort(starts.begin(), starts.end());
  for (uint64_t s : starts) AddEvent(&events, s, 5.0, 2e-7);

  pulse::DeinterleaveSummary out;
  deint.Process(events, &out);
  CHECK(out.tracks_size() == 2);
  CHECK(out.tracks(0).pulse_count() == 9);
  CHECK(out.tracks(1).pulse_count() == 4);
  // PRI estimates get a (tiny) tolerance rather than exact equality:
  // the estimate is a mean of time *differences*, each computed as
  // start/rate - prev/rate, and those divisions round before the
  // subtraction does -- (36/1e7 - 13/1e7) is one ulp off 23/1e7. The
  // pulse counts and track separation above are the integer-exact
  // part; sub-femtosecond PRI rounding is expected behavior.
  CHECK(std::fabs(out.tracks(0).estimated_pri_seconds() - 10.0 / kSampleRateHz) < 1e-18);
  CHECK(std::fabs(out.tracks(1).estimated_pri_seconds() - 23.0 / kSampleRateHz) < 1e-18);
}

// ---------------------------------------------------------------------------
// SpectrogramAnalyzer
// ---------------------------------------------------------------------------

void TestSpectrogramToneLandsInItsBin() {
  // A pure complex tone at bin 3's center frequency correlates to
  // magnitude == amplitude in bin 3 (the rotator cancels the tone's
  // phase exactly), and to near-zero leakage in far-away bins.
  constexpr int kNumBins = 8;
  constexpr double kAmplitude = 3.0;
  const double bin_hz = kSampleRateHz / (2.0 * kNumBins);
  const double tone_hz = (3 + 0.5) * bin_hz;
  const double omega = 2.0 * M_PI * tone_hz / kSampleRateHz;

  pulse::IQBatch batch;
  batch.set_sample_rate_hz(kSampleRateHz);
  batch.set_first_sample_index(0);
  for (int n = 0; n < 1000; ++n) {
    batch.add_i(kAmplitude * std::cos(omega * n));
    batch.add_q(kAmplitude * std::sin(omega * n));
  }

  pulsecore::SpectrogramAnalyzer analyzer(kSampleRateHz, kNumBins);
  pulse::SpectrogramSummary out;
  analyzer.Process(batch, &out);

  CHECK(out.frame_count() == 1);
  CHECK(out.max_magnitude_size() == kNumBins);
  CHECK_EQ_D(out.bin_hz(), bin_hz);
  CHECK(std::fabs(out.max_magnitude(3) - kAmplitude) < 1e-9);
  CHECK(out.max_magnitude(0) < 0.05);
  CHECK(out.max_magnitude(7) < 0.05);
  // mean == max after a single frame.
  CHECK_EQ_D(out.mean_magnitude(3), out.max_magnitude(3));
}

void TestSpectrogramRunningState() {
  // Two batches: tone in bin 2 then silence. max stays at the tone's
  // magnitude; mean halves; frame_count reaches 2.
  constexpr int kNumBins = 8;
  const double bin_hz = kSampleRateHz / (2.0 * kNumBins);
  const double omega = 2.0 * M_PI * ((2 + 0.5) * bin_hz) / kSampleRateHz;

  pulsecore::SpectrogramAnalyzer analyzer(kSampleRateHz, kNumBins);
  pulse::SpectrogramSummary out;

  pulse::IQBatch tone;
  tone.set_sample_rate_hz(kSampleRateHz);
  tone.set_first_sample_index(0);
  for (int n = 0; n < 500; ++n) {
    tone.add_i(2.0 * std::cos(omega * n));
    tone.add_q(2.0 * std::sin(omega * n));
  }
  analyzer.Process(tone, &out);
  const double tone_mag = out.max_magnitude(2);
  CHECK(std::fabs(tone_mag - 2.0) < 1e-9);

  pulse::IQBatch silence = MakeBatch(std::vector<double>(500, 0.0), 500);
  analyzer.Process(silence, &out);
  CHECK(out.frame_count() == 2);
  CHECK_EQ_D(out.max_magnitude(2), tone_mag);
  CHECK_EQ_D(out.mean_magnitude(2), tone_mag / 2.0);
}

// ---------------------------------------------------------------------------
// netutil framing
// ---------------------------------------------------------------------------

void TestFramingRoundTrip() {
  // The transport is a shared-memory ring, so both ends can live in one
  // process: Listen() creates+maps the segment as the consumer end,
  // Connect() maps that same segment as the producer end. Same bytes,
  // same atomics as the cross-process case -- just two Channel handles
  // onto one mapping. Port chosen away from the pipeline's 50052-50055
  // range so a test run can't collide with a live chain.
  const uint16_t port = 59999;
  netutil::Channel* consumer = netutil::Listen(port);
  CHECK(consumer != nullptr);
  // Accept() is a no-op alias for the ring (the segment IS the
  // connection); the services rely on that identity when they Close().
  CHECK(netutil::Accept(consumer) == consumer);
  netutil::Channel* producer = netutil::Connect("127.0.0.1", port);
  CHECK(producer != nullptr);

  // Ordinary payload.
  std::string sent(1000, '\0');
  for (size_t k = 0; k < sent.size(); ++k) sent[k] = static_cast<char>(k % 251);
  CHECK(netutil::SendMessage(producer, sent));
  std::string received;
  CHECK(netutil::RecvMessage(consumer, &received));
  CHECK(received == sent);

  // Empty payload is legal (4-byte length word, zero body).
  CHECK(netutil::SendMessage(producer, std::string()));
  CHECK(netutil::RecvMessage(consumer, &received));
  CHECK(received.empty());

  // A payload the size of the largest frame the pipeline actually
  // produces (~200KB serialized packed iq+events) fits one slot.
  std::string big(200 * 1024, 'x');
  CHECK(netutil::SendMessage(producer, big));
  CHECK(netutil::RecvMessage(consumer, &received));
  CHECK(received == big);

  // A payload no slot can hold is rejected outright rather than
  // truncated -- the ring's one hard capacity limit, documented in
  // framing.h.
  CHECK(!netutil::SendMessage(producer, std::string(600 * 1024, 'y')));

  // Fill every slot, then drain: messages come back in send order.
  for (int k = 0; k < 4; ++k) {
    CHECK(netutil::SendMessage(producer, std::string(1, static_cast<char>('a' + k))));
  }
  for (int k = 0; k < 4; ++k) {
    CHECK(netutil::RecvMessage(consumer, &received));
    CHECK(received == std::string(1, static_cast<char>('a' + k)));
  }

  // Closing the producer raises the writer_done flag; the consumer
  // still drains anything queued ahead of it, then RecvMessage returns
  // false -- how every service detects end-of-stream (the ring's
  // equivalent of reading EOF off a closed socket).
  CHECK(netutil::SendMessage(producer, sent));
  netutil::Close(producer);
  CHECK(netutil::RecvMessage(consumer, &received));
  CHECK(received == sent);
  CHECK(!netutil::RecvMessage(consumer, &received));
  netutil::Close(consumer);  // consumer side unlinks the segment
}

// ---------------------------------------------------------------------------
// AVX2 variant (only when built)
// ---------------------------------------------------------------------------

#ifdef HAVE_AVX_VARIANT
void TestAvxDetectorMatchesScalarExactly() {
  pulsecore::PulseDetector scalar(6.0, kSampleRateHz);
  pulsecore::PulseDetectorAvx avx(6.0, kSampleRateHz);
  pulsecore::SyntheticIQSource source(kSampleRateHz, 2000);
  pulse::IQBatch batch;
  while (source.NextBatch(&batch)) {
    pulse::PulseEventBatch a;
    pulse::PulseEventBatch b;
    scalar.Process(batch, &a);
    avx.Process(batch, &b);
    CHECK(a.SerializeAsString() == b.SerializeAsString());
  }

  // And on a straddling pulse (batch length deliberately not a
  // multiple of 4, so the scalar tail path runs too).
  pulsecore::PulseDetector scalar2(2.0, kSampleRateHz);
  pulsecore::PulseDetectorAvx avx2(2.0, kSampleRateHz);
  pulse::IQBatch first = MakeBatch({0.5, 5.0, 6.0, 5.5, 5.0, 4.0, 3.0}, 0);
  pulse::IQBatch second = MakeBatch({7.0, 4.0, 0.5, 0.5, 0.6}, 7);
  pulse::PulseEventBatch a;
  pulse::PulseEventBatch b;
  scalar2.Process(first, &a);
  scalar2.Process(second, &a);
  avx2.Process(first, &b);
  avx2.Process(second, &b);
  CHECK(a.SerializeAsString() == b.SerializeAsString());
}

void TestAvxJammerMatchesScalarWithinTolerance() {
  pulsecore::JammerDetector scalar(20.0, 0.5);
  pulsecore::JammerDetectorAvx avx(20.0, 0.5);
  pulsecore::SyntheticIQSource source(kSampleRateHz, 2000);
  pulse::IQBatch batch;
  pulse::JamSummary a;
  pulse::JamSummary b;
  while (source.NextBatch(&batch)) {
    scalar.Process(batch, &a);
    avx.Process(batch, &b);
  }
  // Counts and duty cycle are integer-derived: exact. The power sum is
  // a reordered reduction: tolerance (see jammer_avx.h).
  CHECK(a.batches_total() == b.batches_total());
  CHECK(a.batches_flagged() == b.batches_flagged());
  CHECK_EQ_D(a.max_duty_cycle(), b.max_duty_cycle());
  CHECK(std::fabs(a.max_mean_power() - b.max_mean_power()) <=
        1e-9 * std::fabs(a.max_mean_power()));
}
#endif  // HAVE_AVX_VARIANT

}  // namespace

int main() {
  TestIQSourceGoldenValues();
  TestIQSourceDeterminismAndBatching();
  TestDetectorBasicPulse();
  TestDetectorThresholdIsInclusive();
  TestDetectorStraddlesBatchBoundary();
  TestDetectorNoPulses();
  TestStatsAccumulator();
  TestStatsEmpty();
  TestJammerDutyCycle();
  TestDeinterleaverSingleEmitter();
  TestDeinterleaverTwoEmitters();
  TestSpectrogramToneLandsInItsBin();
  TestSpectrogramRunningState();
  TestFramingRoundTrip();
#ifdef HAVE_AVX_VARIANT
  TestAvxDetectorMatchesScalarExactly();
  TestAvxJammerMatchesScalarWithinTolerance();
#endif

  if (failures == 0) {
    std::printf("pulsecore_tests: all tests passed\n");
    return 0;
  }
  std::fprintf(stderr, "pulsecore_tests: %d check(s) FAILED\n", failures);
  return 1;
}
