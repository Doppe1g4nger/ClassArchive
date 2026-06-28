"""Multi-channel (channels, N) IQ handling across the example components."""

import numpy as np
import pytest

from rfdes import HeapScheduler, RFSystem, SignalPayload
from rfdes.components import (
    ADC,
    Amplifier,
    Attenuator,
    Filter,
    Mixer,
    PulseDetector,
    Recorder,
    SpectrumAnalyzer,
    Spectrogrammer,
    Splitter,
)

FS = 10e6


def two_channel(n=256, f0=1e6, f1=2e6):
    k = np.arange(n)
    ch0 = np.exp(2j * np.pi * f0 * k / FS)
    ch1 = 0.5 * np.exp(2j * np.pi * f1 * k / FS)
    return np.stack([ch0, ch1]).astype(np.complex64)  # shape (2, n)


def payload(iq, fc=1e9):
    return SignalPayload(iq, FS, fc)


@pytest.mark.parametrize(
    "comp",
    [
        Amplifier("amp", gain_db=6.0),
        Attenuator("att", loss_db=6.0),
        Splitter("split", loss_db=3.0),
        Mixer("mix", lo_freq=1e6),
        ADC("adc", bits=12, full_scale=2.0),
        Filter("filt", bandwidth=2e6, passband_center=2e6),
    ],
)
def test_processing_components_preserve_channel_shape(comp):
    iq = two_channel()
    out = comp.on_signal(payload(iq))
    assert out.iq.shape == iq.shape  # (2, N) in -> (2, N) out


def test_per_channel_equivalence_for_pointwise_stages():
    """A 2-channel result equals applying the stage to each channel separately."""
    iq = two_channel()
    for comp in (Amplifier("a", 6.0), Mixer("m", 1e6), Filter("f", 2e6, 2e6)):
        out2d = comp.on_signal(payload(iq)).iq
        row0 = comp.on_signal(payload(iq[0])).iq
        row1 = comp.on_signal(payload(iq[1])).iq
        assert np.allclose(out2d[0], row0, atol=1e-5)
        assert np.allclose(out2d[1], row1, atol=1e-5)


def test_pulse_detector_multichannel():
    iq = np.zeros((2, 64), dtype=np.complex64)
    iq[0, 10:20] = 1.0  # burst only on channel 0
    out = PulseDetector("pd", threshold=0.5).on_signal(payload(iq))
    assert out.num_pulses == 1
    assert out.pulses[0, 1] == 10


def test_spectrogrammer_multichannel_is_sum_of_channels():
    iq = two_channel(n=256)
    sg = Spectrogrammer("sg", nfft=64)
    full = sg.on_signal(payload(iq)).power
    p0 = sg.on_signal(payload(iq[0])).power
    p1 = sg.on_signal(payload(iq[1])).power
    assert full.shape == (256 // 64, 64)          # (num_frames, nfft)
    assert np.allclose(full, p0 + p1, atol=1e-4)  # channels summed in power


def test_spectrum_analyzer_multichannel_shape():
    sa = SpectrumAnalyzer("sa")
    sa.on_signal(payload(two_channel(n=128)))
    spec = sa.last_spectrum()
    assert spec.shape == (2, 128)
    # 1-D still works and stays 1-D
    sa1 = SpectrumAnalyzer("sa1")
    sa1.on_signal(payload(np.ones(64, dtype=np.complex64)))
    assert sa1.last_spectrum().shape == (64,)


def test_end_to_end_multichannel_chain_preserves_channels():
    sched = HeapScheduler()
    system = RFSystem(sched)
    lna = system.add(Amplifier("lna", gain_db=10.0))
    mix = system.add(Mixer("mix", lo_freq=1e9))
    adc = system.add(ADC("adc", bits=12, full_scale=50.0))
    rec = system.add(Recorder("rec"))
    lna >> mix >> adc >> rec
    system.set_entry(lna)

    system.on_signal_rx(two_channel(), sample_rate=FS, center_freq=2e9)
    sched.run()
    assert rec.payloads[-1].iq.shape == (2, 256)
