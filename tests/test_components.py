import numpy as np
import pytest

from rfdes import HeapScheduler, RFSystem, SignalPayload
from rfdes.components import ADC, Amplifier, Attenuator, Filter, Mixer, Splitter


def payload(iq, fs=10e6, fc=1e9):
    return SignalPayload(iq=iq.astype(np.complex64), sample_rate=fs, center_freq=fc)


def test_amplifier_scales_by_linear_gain():
    amp = Amplifier("a", gain_db=20.0)  # 20 dB -> x10 in amplitude
    p = amp.on_signal(payload(np.ones(8)))
    assert np.allclose(p.iq, 10.0, atol=1e-4)
    assert p.metadata["gain_db"] == pytest.approx(20.0)


def test_amplifier_accumulates_gain():
    p = payload(np.ones(4))
    p = Amplifier("a", 10.0).on_signal(p)
    p = Amplifier("b", 5.0).on_signal(p)
    assert p.metadata["gain_db"] == pytest.approx(15.0)


def test_attenuator_reduces_amplitude():
    att = Attenuator("att", loss_db=20.0)  # /10 in amplitude
    p = att.on_signal(payload(np.ones(8)))
    assert np.allclose(p.iq, 0.1, atol=1e-4)
    assert p.metadata["gain_db"] == pytest.approx(-20.0)


def test_mixer_shifts_center_frequency():
    mix = Mixer("m", lo_freq=2.5e8)
    p = mix.on_signal(payload(np.ones(16), fc=1e9))
    assert p.center_freq == pytest.approx(0.75e9)


def test_mixer_downconverts_tone_to_baseband():
    fs = 10e6
    f_in = 2e6
    n = np.arange(2048)
    iq = np.exp(2j * np.pi * f_in * n / fs)
    mix = Mixer("m", lo_freq=f_in)
    out = mix.on_signal(payload(iq, fs=fs))
    # after mixing by the same frequency, the tone lands at DC (bin 0)
    spec = np.abs(np.fft.fft(out.iq))
    assert int(np.argmax(spec)) == 0


def test_splitter_passthrough_preserves_iq():
    sp = Splitter("s")
    iq = np.arange(8) + 1j * np.arange(8)
    p = sp.on_signal(payload(iq))
    assert np.allclose(p.iq, iq)


def test_splitter_with_loss():
    sp = Splitter("s", loss_db=6.0)  # ~x0.5
    p = sp.on_signal(payload(np.ones(8)))
    assert np.allclose(np.abs(p.iq), 10 ** (-6.0 / 20), atol=1e-4)


def test_adc_quantizes_to_levels():
    adc = ADC("adc", bits=2, full_scale=1.0)  # step = 2/4 = 0.5
    iq = np.array([0.1 + 0.1j, 0.4 + 0.0j, 0.9 + 0.9j])
    p = adc.on_signal(payload(iq))
    # values land on multiples of the 0.5 step
    for v in p.iq:
        assert v.real % 0.5 == pytest.approx(0.0, abs=1e-6) or v.real == pytest.approx(round(v.real / 0.5) * 0.5)
    assert p.metadata["adc_bits"] == 2


def test_adc_clips_to_full_scale():
    adc = ADC("adc", bits=12, full_scale=1.0)
    p = adc.on_signal(payload(np.array([5.0 + 5.0j])))
    assert p.iq[0].real <= 1.0 + 1e-6
    assert p.iq[0].imag <= 1.0 + 1e-6


def test_filter_rejects_out_of_band_tone():
    fs = 10e6
    n = np.arange(4096)
    # in-band tone at 0.5 MHz, out-of-band tone at 4 MHz
    inband = np.exp(2j * np.pi * 0.5e6 * n / fs)
    outband = np.exp(2j * np.pi * 4e6 * n / fs)
    filt = Filter("f", bandwidth=2e6)  # pass +/-1 MHz around baseband
    out = filt.on_signal(payload(inband + outband, fs=fs))
    spec = np.abs(np.fft.fft(out.iq))
    freqs = np.fft.fftfreq(n.size, d=1.0 / fs)
    inband_power = spec[np.argmin(np.abs(freqs - 0.5e6))]
    outband_power = spec[np.argmin(np.abs(freqs - 4e6))]
    assert inband_power > 100 * (outband_power + 1e-9)


def test_full_chain_latency_equals_sum_of_delays():
    sched = HeapScheduler()
    system = RFSystem(sched)
    from rfdes.components import Recorder

    a = system.add(Amplifier("a", 10.0, processing_delay=1.5))
    m = system.add(Mixer("m", 1e9, processing_delay=2.5))
    rec = system.add(Recorder("rec"))
    a >> m >> rec
    system.set_entry(a)
    system.on_signal_rx(np.ones(8, dtype=np.complex64), 1e6, 2e9, t=0.0)
    sched.run()
    assert sched.now() == pytest.approx(1.5 + 2.5)
