"""Duty-cycle-based jamming detector -- Python port of
common/src/jammer.cpp / common/include/jammer.h.
"""
from pulsecore import pulse_pb2


class JammerDetector:
    def __init__(self, power_threshold: float, duty_cycle_threshold: float):
        self._power_threshold = power_threshold
        self._duty_cycle_threshold = duty_cycle_threshold
        self._batches_total = 0
        self._batches_flagged = 0
        self._max_duty_cycle = 0.0
        self._max_mean_power = 0.0

    def process(self, batch: "pulse_pb2.IQBatch", out: "pulse_pb2.JamSummary") -> None:
        n = len(batch.i)
        if n > 0:
            power_threshold = self._power_threshold
            power_sum = 0.0
            over_threshold = 0
            # Packed columnar layout: zip over the two packed arrays,
            # no per-sample message access.
            for si, sq in zip(batch.i, batch.q):
                power = si * si + sq * sq
                power_sum += power
                if power >= power_threshold:
                    over_threshold += 1

            mean_power = power_sum / n
            duty_cycle = over_threshold / n

            self._batches_total += 1
            if duty_cycle >= self._duty_cycle_threshold:
                self._batches_flagged += 1
            if duty_cycle > self._max_duty_cycle:
                self._max_duty_cycle = duty_cycle
            if mean_power > self._max_mean_power:
                self._max_mean_power = mean_power

        out.Clear()
        out.batches_total = self._batches_total
        out.batches_flagged = self._batches_flagged
        out.max_duty_cycle = self._max_duty_cycle
        out.max_mean_power = self._max_mean_power
