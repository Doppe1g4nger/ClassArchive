import numpy as np
import pytest

from rfdes import (
    Component,
    DataObject,
    HeapScheduler,
    MergeComponent,
    RFSystem,
    SignalPayload,
)
from rfdes.components import (
    Amplifier,
    DetectionFusion,
    PulseDetector,
    Recorder,
    Spectrogrammer,
    Splitter,
)
from rfdes.datatypes import DetectionReport, PulseBuffer, Spectrogram


# -- a tiny merge component over a single data type, for focused tests --------
class TwoInMerge(MergeComponent):
    inputs = {"a": (SignalPayload,), "b": (SignalPayload,)}
    produces = SignalPayload

    def on_merge(self, inputs):
        merged_iq = inputs["a"].iq + inputs["b"].iq
        return SignalPayload(
            iq=merged_iq,
            sample_rate=inputs["a"].sample_rate,
            center_freq=inputs["a"].center_freq,
            start_time=max(inputs["a"].start_time, inputs["b"].start_time),
        )


def payload(val, n=4):
    return SignalPayload(np.full(n, val, dtype=np.complex64), 1e6, 1e9)


def test_merge_fires_only_after_all_ports_filled():
    sched = HeapScheduler()
    system = RFSystem(sched)
    merge = system.add(TwoInMerge("merge"))
    rec = system.add(Recorder("rec"))
    merge.subscribe(rec)

    merge.receive(payload(1.0), port="a")
    assert rec.payloads == []  # only one port filled -> no firing

    merge.receive(payload(2.0), port="b")
    sched.run()
    assert len(rec.payloads) == 1
    assert np.allclose(rec.payloads[0].iq, 3.0)  # 1 + 2


def test_merge_fifo_zip_pairs_in_order():
    sched = HeapScheduler()
    system = RFSystem(sched)
    merge = system.add(TwoInMerge("merge"))
    rec = system.add(Recorder("rec"))
    merge.subscribe(rec)

    merge.receive(payload(1.0), port="a")
    merge.receive(payload(10.0), port="a")
    merge.receive(payload(2.0), port="b")  # pairs with first a -> 3
    merge.receive(payload(20.0), port="b")  # pairs with second a -> 30
    sched.run()
    assert [float(p.iq[0].real) for p in rec.payloads] == [3.0, 30.0]


def test_unknown_port_on_merge_raises():
    sched = HeapScheduler()
    system = RFSystem(sched)
    merge = system.add(TwoInMerge("merge"))
    with pytest.raises(KeyError):
        merge.receive(payload(1.0), port="zzz")


class KeyedMerge(MergeComponent):
    inputs = {"a": (SignalPayload,), "b": (SignalPayload,)}
    produces = SignalPayload

    def correlation_key(self, data):
        return data.metadata.get("id")

    def on_merge(self, inputs):
        return SignalPayload(
            iq=inputs["a"].iq + inputs["b"].iq,
            sample_rate=1e6,
            center_freq=1e9,
            metadata={"id": inputs["a"].metadata["id"]},
        )


def keyed(val, key):
    return SignalPayload(np.full(4, val, dtype=np.complex64), 1e6, 1e9, metadata={"id": key})


def test_correlation_key_matches_by_key_not_arrival_order():
    sched = HeapScheduler()
    system = RFSystem(sched)
    merge = system.add(KeyedMerge("merge"))
    rec = system.add(Recorder("rec"))
    merge.subscribe(rec)

    merge.receive(keyed(1.0, "x"), port="a")
    merge.receive(keyed(2.0, "y"), port="a")
    # b arrives for "y" first: must pair with the "y" item on port a, not the
    # earlier "x" item.
    merge.receive(keyed(20.0, "y"), port="b")
    sched.run()
    assert len(rec.payloads) == 1
    assert rec.payloads[0].metadata["id"] == "y"
    assert np.allclose(rec.payloads[0].iq, 22.0)  # 2 + 20


def test_detection_fusion_end_to_end():
    sched = HeapScheduler()
    system = RFSystem(sched)
    lna = system.add(Amplifier("lna", gain_db=0.0))
    split = system.add(Splitter("split"))
    pd = system.add(PulseDetector("pd", threshold=0.5))
    sp = system.add(Spectrogrammer("sp", nfft=64))
    fusion = system.add(DetectionFusion("fusion"))
    rec = system.add(Recorder("rec"))
    lna.subscribe(split)
    split.subscribe(pd)
    split.subscribe(sp)
    pd.subscribe(fusion, port="pulses")
    sp.subscribe(fusion, port="spectrogram")
    fusion.subscribe(rec)
    system.set_entry(lna)

    fs = 10e6
    n = np.arange(256)
    env = np.full(256, 0.05)
    env[50:90] = 1.0  # one burst
    iq = (env * np.exp(2j * np.pi * 1e6 * n / fs)).astype(np.complex64)
    system.on_signal_rx(iq, sample_rate=fs, center_freq=2.4e9)
    sched.run()

    assert len(rec.payloads) == 1
    report = rec.payloads[0]
    assert isinstance(report, DetectionReport)
    assert report.fields["num_pulses"] == 1


def test_intermediate_data_types_are_correct():
    pd = PulseDetector("pd", threshold=0.5)
    sp = Spectrogrammer("sp", nfft=32)
    iq = np.zeros(64, dtype=np.complex64)
    iq[10:20] = 1.0
    sig = SignalPayload(iq, 1e6, 1e9)
    pulses = pd.on_signal(sig)
    spectro = sp.on_signal(sig)
    assert isinstance(pulses, PulseBuffer)
    assert pulses.num_pulses == 1
    assert pulses.pulses[0, 1] == 10  # width
    assert isinstance(spectro, Spectrogram)
    assert spectro.shape[1] == 32
