"""Unit tests for python/pulsecore/, mirroring tests/pulsecore_tests.cpp
case for case. The golden values in TestIQSource are the *same doubles*
the C++ suite asserts -- both suites pinning one set of constants is
what turns "the two generators are bit-identical" from a claim verified
occasionally by hand into one enforced on every test run.
"""
import math
import socket
import struct
import unittest

from tests import *  # noqa: F401,F403 -- path bootstrap (see __init__.py)

from pulsecore import pulse_pb2
from pulsecore.iq_source import SyntheticIQSource
from pulsecore.pulse_detector import PulseDetector
from pulsecore.pulse_stats import PulseStatsAccumulator
from pulsecore.jammer import JammerDetector
from pulsecore.deinterleaver import Deinterleaver
from pulsecore.spectrogram import SpectrogramAnalyzer
from microservice import framing

SAMPLE_RATE_HZ = 10_000_000.0


def make_batch(amplitudes, start_index):
    """Sample k gets i=amplitudes[k], q=0 -- magnitude is exactly
    amplitudes[k], keeping expected outputs computable by hand."""
    batch = pulse_pb2.IQBatch()
    batch.sample_rate_hz = SAMPLE_RATE_HZ
    batch.first_sample_index = start_index
    batch.i.extend(amplitudes)
    batch.q.extend([0.0] * len(amplitudes))
    return batch


def add_event(batch, start, peak, duration_s):
    batch.start_sample.append(start)
    batch.end_sample.append(start + 1)
    batch.peak_amplitude.append(peak)
    batch.mean_amplitude.append(peak)
    batch.duration_seconds.append(duration_s)


class TestIQSource(unittest.TestCase):
    def test_golden_values(self):
        # Pinned from the generator and asserted identically by the C++
        # suite (tests/pulsecore_tests.cpp) -- if either language's
        # generator drifts, its own unit tests fail immediately.
        src = SyntheticIQSource(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=10)
        batch = pulse_pb2.IQBatch()
        self.assertTrue(src.next_batch(batch))
        self.assertEqual(len(batch.i), 108)  # 10 * 10 + 8 leading gap
        self.assertEqual(len(batch.q), 108)
        self.assertEqual(batch.sample_rate_hz, SAMPLE_RATE_HZ)

        self.assertEqual(batch.first_sample_index, 0)
        self.assertEqual(batch.i[0], -0.4973561074578567)
        self.assertEqual(batch.q[0], 0.16031197753276494)
        self.assertEqual(batch.i[7], -0.044113119725164296)
        self.assertEqual(batch.q[7], -0.2474199623678392)
        self.assertEqual(batch.i[8], 6.697328498560646)
        self.assertEqual(batch.q[8], 6.5812648301410235)
        self.assertEqual(batch.i[107], -0.31797348238014933)
        self.assertEqual(batch.q[107], 0.484969302077072)

        self.assertFalse(src.next_batch(batch))

    def test_determinism_and_batching(self):
        a = SyntheticIQSource(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=50000)
        b = SyntheticIQSource(sample_rate_hz=SAMPLE_RATE_HZ, num_pulses=50000)
        ba = pulse_pb2.IQBatch()
        bb = pulse_pb2.IQBatch()
        batches = 0
        samples = 0
        while a.next_batch(ba):
            self.assertTrue(b.next_batch(bb))
            self.assertEqual(ba.SerializeToString(), bb.SerializeToString())
            samples += len(ba.i)
            batches += 1
        self.assertFalse(b.next_batch(bb))
        self.assertEqual(batches, 51)
        self.assertEqual(samples, 500008)


class TestPulseDetector(unittest.TestCase):
    def test_basic_pulse(self):
        detector = PulseDetector(amplitude_threshold=2.0, sample_rate_hz=SAMPLE_RATE_HZ)
        batch = make_batch([0.5, 0.5, 5.0, 7.0, 6.0, 0.5, 0.5], 100)
        out = pulse_pb2.PulseEventBatch()
        detector.process(batch, out)

        self.assertEqual(len(out.start_sample), 1)
        self.assertEqual(out.start_sample[0], 102)
        self.assertEqual(out.end_sample[0], 105)
        self.assertEqual(out.peak_amplitude[0], 7.0)
        self.assertEqual(out.mean_amplitude[0], 6.0)
        self.assertEqual(out.duration_seconds[0], 3.0 / SAMPLE_RATE_HZ)

    def test_threshold_is_inclusive(self):
        detector = PulseDetector(amplitude_threshold=2.0, sample_rate_hz=SAMPLE_RATE_HZ)
        batch = make_batch([0.0, 2.0, 0.0], 0)
        out = pulse_pb2.PulseEventBatch()
        detector.process(batch, out)
        self.assertEqual(len(out.start_sample), 1)
        self.assertEqual(out.peak_amplitude[0], 2.0)

    def test_pulse_straddles_batch_boundary(self):
        # Never produced by the synthetic signal at this repo's
        # constants, but explicitly supported: state carries across
        # process() calls.
        detector = PulseDetector(amplitude_threshold=2.0, sample_rate_hz=SAMPLE_RATE_HZ)
        out = pulse_pb2.PulseEventBatch()
        detector.process(make_batch([0.5, 5.0, 6.0], 0), out)
        self.assertEqual(len(out.start_sample), 0)  # still open at batch end
        detector.process(make_batch([7.0, 4.0, 0.5, 0.5], 3), out)
        self.assertEqual(len(out.start_sample), 1)
        self.assertEqual(out.start_sample[0], 1)
        self.assertEqual(out.end_sample[0], 5)
        self.assertEqual(out.peak_amplitude[0], 7.0)
        self.assertEqual(out.mean_amplitude[0], (5.0 + 6.0 + 7.0 + 4.0) / 4.0)
        self.assertEqual(out.duration_seconds[0], 4.0 / SAMPLE_RATE_HZ)

    def test_no_pulses(self):
        detector = PulseDetector(amplitude_threshold=2.0, sample_rate_hz=SAMPLE_RATE_HZ)
        out = pulse_pb2.PulseEventBatch()
        detector.process(make_batch([0.5, 1.0, 1.9, 0.1], 0), out)
        self.assertEqual(len(out.start_sample), 0)


class TestPulseStats(unittest.TestCase):
    def test_accumulation_across_batches(self):
        acc = PulseStatsAccumulator(sample_rate_hz=SAMPLE_RATE_HZ)

        batch1 = pulse_pb2.PulseEventBatch()
        add_event(batch1, 0, 5.0, 2e-7)
        add_event(batch1, 10, 7.0, 2e-7)
        acc.add(batch1)

        # PRI state must carry across add() calls.
        batch2 = pulse_pb2.PulseEventBatch()
        add_event(batch2, 30, 3.0, 4e-7)
        acc.add(batch2)

        s = acc.finalize()
        self.assertEqual(s.pulse_count, 3)
        self.assertEqual(s.mean_peak_amplitude, (5.0 + 7.0 + 3.0) / 3.0)
        self.assertEqual(s.min_peak_amplitude, 3.0)
        self.assertEqual(s.max_peak_amplitude, 7.0)
        self.assertEqual(s.mean_duration_seconds, (2e-7 + 2e-7 + 4e-7) / 3.0)
        self.assertEqual(
            s.mean_pri_seconds,
            ((10.0 / SAMPLE_RATE_HZ) + (20.0 / SAMPLE_RATE_HZ)) / 2.0,
        )

    def test_empty(self):
        acc = PulseStatsAccumulator(sample_rate_hz=SAMPLE_RATE_HZ)
        s = acc.finalize()
        self.assertEqual(s.pulse_count, 0)
        self.assertEqual(s.mean_peak_amplitude, 0.0)
        self.assertEqual(s.mean_pri_seconds, 0.0)


class TestJammer(unittest.TestCase):
    def test_duty_cycle_and_running_maxima(self):
        jammer = JammerDetector(power_threshold=2.0, duty_cycle_threshold=0.5)
        out = pulse_pb2.JamSummary()

        hot = make_batch([2.0, 3.0, 1.5, 0.1], 0)  # powers 4, 9, 2.25, 0.01
        jammer.process(hot, out)
        self.assertEqual(out.batches_total, 1)
        self.assertEqual(out.batches_flagged, 1)
        self.assertEqual(out.max_duty_cycle, 0.75)
        self.assertEqual(out.max_mean_power, (4.0 + 9.0 + 2.25 + 0.01) / 4.0)

        cool = make_batch([2.0, 0.1, 0.1, 0.1], 4)
        jammer.process(cool, out)
        self.assertEqual(out.batches_total, 2)
        self.assertEqual(out.batches_flagged, 1)
        self.assertEqual(out.max_duty_cycle, 0.75)  # persists from batch 1
        self.assertEqual(out.max_mean_power, (4.0 + 9.0 + 2.25 + 0.01) / 4.0)


class TestDeinterleaver(unittest.TestCase):
    def test_single_emitter(self):
        deint = Deinterleaver(sample_rate_hz=SAMPLE_RATE_HZ, pri_tolerance_seconds=1e-7)
        events = pulse_pb2.PulseEventBatch()
        for k in range(5):
            add_event(events, 10 * k, 5.0, 2e-7)
        out = pulse_pb2.DeinterleaveSummary()
        deint.process(events, out)
        self.assertEqual(len(out.tracks), 1)
        self.assertEqual(out.tracks[0].pulse_count, 5)
        self.assertEqual(out.tracks[0].estimated_pri_seconds, 10.0 / SAMPLE_RATE_HZ)
        self.assertEqual(out.tracks[0].mean_peak_amplitude, 5.0)

    def test_two_emitters(self):
        # See tests/pulsecore_tests.cpp's TestDeinterleaverTwoEmitters
        # for why emitter A establishes its PRI before B appears: two
        # cold-start interleaved trains would get their seed pulses
        # merged -- a documented limitation of the simplified
        # sequential-PRI algorithm, not a regression this test should
        # hide or accidentally depend on.
        deint = Deinterleaver(sample_rate_hz=SAMPLE_RATE_HZ, pri_tolerance_seconds=1e-7)
        starts = sorted([10 * k for k in range(9)] + [13 + 23 * k for k in range(4)])
        events = pulse_pb2.PulseEventBatch()
        for s in starts:
            add_event(events, s, 5.0, 2e-7)
        out = pulse_pb2.DeinterleaveSummary()
        deint.process(events, out)
        self.assertEqual(len(out.tracks), 2)
        self.assertEqual(out.tracks[0].pulse_count, 9)
        self.assertEqual(out.tracks[1].pulse_count, 4)
        # Same tolerance rationale as the C++ test: the PRI estimate is
        # a mean of differences of already-rounded divisions.
        self.assertAlmostEqual(
            out.tracks[0].estimated_pri_seconds, 10.0 / SAMPLE_RATE_HZ, delta=1e-18
        )
        self.assertAlmostEqual(
            out.tracks[1].estimated_pri_seconds, 23.0 / SAMPLE_RATE_HZ, delta=1e-18
        )


class TestSpectrogram(unittest.TestCase):
    NUM_BINS = 8

    def _tone_batch(self, bin_index, amplitude, n, start_index=0):
        bin_hz = SAMPLE_RATE_HZ / (2.0 * self.NUM_BINS)
        omega = 2.0 * math.pi * ((bin_index + 0.5) * bin_hz) / SAMPLE_RATE_HZ
        batch = pulse_pb2.IQBatch()
        batch.sample_rate_hz = SAMPLE_RATE_HZ
        batch.first_sample_index = start_index
        for k in range(n):
            idx = start_index + k
            batch.i.append(amplitude * math.cos(omega * idx))
            batch.q.append(amplitude * math.sin(omega * idx))
        return batch

    def test_tone_lands_in_its_bin(self):
        analyzer = SpectrogramAnalyzer(sample_rate_hz=SAMPLE_RATE_HZ, num_bins=self.NUM_BINS)
        out = pulse_pb2.SpectrogramSummary()
        analyzer.process(self._tone_batch(3, 3.0, 1000), out)

        self.assertEqual(out.frame_count, 1)
        self.assertEqual(len(out.max_magnitude), self.NUM_BINS)
        self.assertEqual(out.bin_hz, SAMPLE_RATE_HZ / (2.0 * self.NUM_BINS))
        self.assertLess(abs(out.max_magnitude[3] - 3.0), 1e-9)
        self.assertLess(out.max_magnitude[0], 0.05)
        self.assertLess(out.max_magnitude[7], 0.05)
        self.assertEqual(out.mean_magnitude[3], out.max_magnitude[3])

    def test_running_state(self):
        analyzer = SpectrogramAnalyzer(sample_rate_hz=SAMPLE_RATE_HZ, num_bins=self.NUM_BINS)
        out = pulse_pb2.SpectrogramSummary()
        analyzer.process(self._tone_batch(2, 2.0, 500), out)
        tone_mag = out.max_magnitude[2]
        self.assertLess(abs(tone_mag - 2.0), 1e-9)

        analyzer.process(make_batch([0.0] * 500, 500), out)
        self.assertEqual(out.frame_count, 2)
        self.assertEqual(out.max_magnitude[2], tone_mag)
        self.assertEqual(out.mean_magnitude[2], tone_mag / 2.0)


class TestFraming(unittest.TestCase):
    def test_round_trip(self):
        a, b = socket.socketpair()
        try:
            payload = bytes(k % 251 for k in range(1000))
            self.assertTrue(framing.send_message(a, payload))
            self.assertEqual(framing.recv_message(b), payload)

            # Empty payload is legal: 4-byte header, zero body.
            self.assertTrue(framing.send_message(a, b""))
            self.assertEqual(framing.recv_message(b), b"")
        finally:
            a.close()
            b.close()

    def test_wire_format_is_4_byte_big_endian_prefix(self):
        # The framing layer IS the wire contract with the C++ services
        # (README: "a capture of this traffic is indistinguishable from
        # the C++ build's"), so test the bytes, not just the round trip.
        a, b = socket.socketpair()
        try:
            framing.send_message(a, b"abc")
            raw = b.recv(7, socket.MSG_WAITALL)
            self.assertEqual(raw, struct.pack(">I", 3) + b"abc")
        finally:
            a.close()
            b.close()

    def test_peer_close_ends_stream(self):
        a, b = socket.socketpair()
        try:
            framing.send_message(a, b"last")
            a.close()
            self.assertEqual(framing.recv_message(b), b"last")
            self.assertIsNone(framing.recv_message(b))
        finally:
            b.close()


if __name__ == "__main__":
    unittest.main()
