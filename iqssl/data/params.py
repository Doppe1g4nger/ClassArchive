"""Generator configuration, priors, and the difficulty presets.

Everything that decides *what the task is* lives here, so the difficulty of the
benchmark is a small readable table rather than an emergent property of code
scattered across the generator.

Several bounds below are not free parameters. They were derived by running the
difficulty gate (:mod:`iqssl.cli.difficulty_report`), watching it fail, and
diagnosing why; each one carries the reasoning that fixed it. Widening them is
how the task quietly becomes trivial or impossible, so they are commented rather
than merely written down.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from iqssl.dsp.filters import DEFAULT_SPAN, DEFAULT_SPS

GENERATOR_VERSION = "v1"
"""Bumped whenever generation semantics change. Folded into the dataset hash so a
semantic change invalidates it even if the bytes coincidentally matched."""

SAMPLE_RATE_HZ = 1e6
"""Nominal sample rate. Only phase-noise linewidth and Doppler are expressed in
absolute Hz; everything else is normalized, so this is a labelling convention."""

FINGERPRINT_SNR_FLOOR_DB = 12.0
"""Below roughly this SNR, RF fingerprinting stops working.

Domain knowledge, not a measurement of this generator. The emitter label lives in
small amplitude and phase distortions -- IQ imbalance, DC offset, PA compression
-- and once the noise floor rises to their scale there is nothing left to read.
Modulation class survives far lower, which is why the two axes diverge.

This is the constraint the difficulty ladder was originally built in violation
of. `medium` spanned 0-20 dB and `hard` -5-15 dB, so most of their mass sat below
the floor: the oracle reached 0.436 and 0.149 against a 0.0625 chance line, and
`hard`'s *best* SNR quartile (10-15 dB) straddled the knee. The ladder was being
made harder by removing the signal rather than by obscuring it, which is a
different thing and not the one the benchmark wants to measure.

Difficulty now comes from multipath, which `PRESETS` records as the dominant
destroyer of the fingerprint, while SNR stays mostly above the floor with a
deliberate tail below it so the knee itself remains observable.
"""

ALL_MODULATIONS = (
    "bpsk",
    "qpsk",
    "psk8",
    "qam16",
    "qam64",
    "pam4",
    "ook",
    "gfsk",
    "cpfsk",
    "msk",
)

CPM_MODULATIONS = frozenset({"gfsk", "cpfsk", "msk"})

ROLLOFF_CHOICES = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
"""RRC rolloff is drawn from a *discrete* set rather than a continuous prior.

Signals sharing a rolloff share their pulse-shaping taps, which is what lets the
generator shape a whole group in one batched convolution. A continuous prior
would force one convolution per sample and make generation roughly two orders of
magnitude slower for no gain in realism -- six values already exceed anything a
model could exploit as a shortcut.
"""

NUISANCE_FIELDS: tuple[str, ...] = (
    "snr_db_nominal",
    "cfo_norm",
    "sro_ppm",
    "timing_offset_samples",
    "phase0_rad",
    "delay_spread_symbols",
)
"""The continuous nuisance vector, and the regression target for "what did the
representation discard?".

Every field here is drawn for *every* sample in *every* preset. That is a
requirement, not a coincidence: the stored standardization divides by each
field's train-split standard deviation, so a field that is constant under some
preset would divide by zero. It is also why ``rolloff`` is absent despite being
drawn per sample -- continuous-phase modulations are never pulse-shaped, so
rolloff is not applied to three of the ten classes and would be a meaningless
entry in their nuisance vector.
"""


@dataclass(frozen=True)
class EmitterPrior:
    """Priors over the per-device impairments that constitute the fingerprint.

    Drawn once per emitter and then held fixed, so together they are the identity
    a fingerprinting model has to recover.
    """

    iq_gain_db: tuple[float, float] = (-2.5, 2.5)
    iq_phase_deg: tuple[float, float] = (-15.0, 15.0)

    dc_dbc: tuple[float, float] = (-28.0, -14.0)
    """LO leakage relative to buffer RMS.

    The lower bound is a hard physical floor, not a taste. A buffer's own sample
    mean has magnitude ~``1/sqrt(L)``; at ``L = 1024`` that is -30 dBc. An offset
    below that is buried in the signal's own mean and is unrecoverable by *any*
    estimator, so widening this bound does not make the task harder, it makes a
    fraction of the emitter label pure noise.
    """

    pa_ibo_db: tuple[float, float] = (2.0, 10.0)
    """Saleh input back-off. Low back-off means hard compression.

    These spreads were widened once, from (+/-1.2 dB, +/-9 deg, -30..-19 dBc,
    3-11 dB), after the gate measured a supervised oracle at 0.63 against its
    0.85-0.95 band -- the fingerprint was present at ten times chance but not
    fully resolvable in 1024 samples. Widening is normally the *last* knob to
    reach for, because it is the fastest way to make the task trivially easy;
    it was the right one here only because the classical baseline sat at 0.139
    against a 0.60 ceiling, leaving room to spend. Check that headroom before
    touching these again.
    """

    pn_linewidth_hz: tuple[float, float] = (15.0, 120.0)
    """Oscillator linewidth, bounded **above** at 120 Hz.

    Phase noise is a Wiener process, so accumulated drift over a buffer is
    ``sqrt(2*pi*lw*L/fs)``. At 800 Hz that is 2.3 rad over 1024 samples, which
    randomizes the imbalance axis and smears LO leakage into the noise floor --
    it erases every phase-coherent part of the fingerprint and leaves only the PA
    envelope. At 120 Hz it is 0.88 rad: the oscillator still has a distinguishing
    spectral character, but the rest of the signature survives it.
    """

    cfo_bias_hz: tuple[float, float] = (0.0, 0.0)
    """Per-emitter *systematic* carrier offset. Zero by default, and that matters.

    A fixed CFO bias per device would correlate a channel nuisance with emitter
    identity, and the nuisance analysis rests on those being independent. It is
    kept as a knob only so the confound can be introduced deliberately, as an
    ablation demonstrating what it costs.
    """


@dataclass(frozen=True)
class ChannelPrior:
    """Priors over the propagation nuisances, resampled per buffer."""

    snr_db: tuple[float, float] = (0.0, 20.0)
    cfo_norm: tuple[float, float] = (-1e-3, 1e-3)
    """Cycles per sample."""

    sro_ppm: tuple[float, float] = (-40.0, 40.0)
    timing_offset_samples: tuple[float, float] = (-0.5, 0.5)
    n_taps_choices: tuple[int, ...] = (1,)
    delay_spread_symbols: tuple[float, float] = (0.3, 1.6)
    k_factor_db: float = -100.0
    """Rician K. Very negative means pure Rayleigh, no specular component."""


@dataclass(frozen=True)
class Preset:
    """One rung of the difficulty ladder."""

    name: str
    store_len: int
    """Stored buffer length. Longer than ``crop_len`` so random cropping has room
    to move without ever reaching the filter transients trimmed at generation."""

    crop_len: int
    n_emitters: int
    emitter: EmitterPrior = field(default_factory=EmitterPrior)
    channel: ChannelPrior = field(default_factory=ChannelPrior)

    @property
    def spans_0db(self) -> bool:
        lo, hi = self.channel.snr_db
        return lo <= 3.0 and hi >= -3.0


# The ladder below was calibrated by measurement, not by taste. Holding the
# emitter prior fixed and varying one channel knob at a time, oracle accuracy on
# emitter id moved like this:
#
#     clean, high SNR, single tap ............. 0.42  (baseline)
#     + carrier frequency offset .............. 0.45  (no cost -- CFO is cheap)
#     + 2-tap multipath ....................... 0.22  (roughly halves it)
#     + low SNR ............................... 0.10  (nearly kills it)
#
# So multipath is the dominant destroyer of the fingerprint, low SNR is second,
# and CFO costs essentially nothing. When a preset lands outside its band, reach
# for those in that order. Widening the *emitter* spreads is the last resort: it
# is also the fastest way to make the task trivially easy again.

PRESETS: dict[str, Preset] = {
    "smoke": Preset(
        name="smoke",
        store_len=320,
        crop_len=256,
        n_emitters=8,
        channel=ChannelPrior(snr_db=(15.0, 30.0), n_taps_choices=(1,)),
    ),
    "easy": Preset(
        name="easy",
        store_len=1280,
        crop_len=1024,
        n_emitters=16,
        channel=ChannelPrior(
            snr_db=(15.0, 30.0),
            cfo_norm=(-5e-4, 5e-4),
            sro_ppm=(-20.0, 20.0),
            n_taps_choices=(1,),
        ),
    ),
    # `medium` and `hard` carry their difficulty in *multipath*, not in SNR.
    # Their original ranges (0-20 and -5-15) put most of their mass below
    # FINGERPRINT_SNR_FLOOR_DB, where the emitter label does not survive at all:
    # gated at 48k training buffers the oracle reached 0.436 and 0.149 against
    # 0.0625 chance, with train accuracy 1.000 in both cases. The floors below
    # sit just under the knee so a sub-threshold population still exists to
    # measure -- that is what `supervised_cnn_sub_threshold` checks -- while the
    # bulk of each preset stays in the regime where the task is possible.
    "medium": Preset(
        name="medium",
        store_len=1280,
        crop_len=1024,
        n_emitters=16,
        channel=ChannelPrior(
            snr_db=(10.0, 25.0),
            cfo_norm=(-1e-3, 1e-3),
            sro_ppm=(-40.0, 40.0),
            n_taps_choices=(1, 2, 3),
        ),
    ),
    "hard": Preset(
        name="hard",
        store_len=1280,
        crop_len=1024,
        n_emitters=16,
        channel=ChannelPrior(
            snr_db=(8.0, 20.0),
            cfo_norm=(-2e-3, 2e-3),
            sro_ppm=(-60.0, 60.0),
            n_taps_choices=(2, 3, 4),
            delay_spread_symbols=(0.5, 2.5),
        ),
    ),
}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown difficulty {name!r}; options: {sorted(PRESETS)}") from None


@dataclass(frozen=True)
class GeneratorConfig:
    """Everything that determines the dataset's identity.

    Frozen and hashed into the manifest: two datasets with the same config and
    seed are the same dataset, and the aggregator refuses to pool runs whose
    dataset hashes disagree.
    """

    n_samples: int = 200_000
    difficulty: str = "medium"
    seed: int = 0
    shard_size: int = 20_000
    span: int = DEFAULT_SPAN
    sps: int = DEFAULT_SPS
    normalize: str = "rms"
    modulations: tuple[str, ...] = ALL_MODULATIONS

    def __post_init__(self) -> None:
        get_preset(self.difficulty)  # fail loudly at construction, not mid-build
        unknown = set(self.modulations) - set(ALL_MODULATIONS)
        if unknown:
            raise ValueError(
                f"unknown modulations {sorted(unknown)}; options: {list(ALL_MODULATIONS)}"
            )
        if not self.modulations:
            raise ValueError("need at least one modulation")
        if self.span % 2 == 0:
            raise ValueError(f"span must be odd so the RRC delay is integral, got {self.span}")

    @property
    def preset(self) -> Preset:
        return get_preset(self.difficulty)

    def identity(self) -> dict:
        """The subset of the config that defines the *content* of the dataset.

        ``shard_size`` is excluded on purpose: how many files the samples are
        written across is a storage decision, and an otherwise identical dataset
        must not look like a different one to the aggregator because someone
        chose a different shard size.
        """
        d = asdict(replace(self, shard_size=0))
        d.pop("shard_size")
        d["modulations"] = list(self.modulations)
        d["preset"] = asdict(self.preset)
        return d


def n_symbols_for(store_len: int, sps: int, span: int) -> int:
    """Symbols to generate so that, after trimming filter transients, at least
    ``store_len`` clean samples remain.

    The RRC runs in ``mode='full'`` and is trimmed deliberately rather than
    centred, because leaving a ramp-up at a fixed offset in every buffer is a
    shortcut feature a network finds immediately.
    """
    transient = span * sps
    return math.ceil((store_len + 2 * transient) / sps) + span


@dataclass(frozen=True)
class Emitter:
    """One device's realized fingerprint. Fixed for the life of the dataset."""

    emitter_id: int
    iq_gain_db: float
    iq_phase_deg: float
    dc_i_dbc: float
    dc_q_dbc: float
    pa_ibo_db: float
    pn_linewidth_hz: float
    cfo_bias_hz: float

    def to_dict(self) -> dict:
        return asdict(self)


def draw_emitters(prior: EmitterPrior, n: int, seed: int) -> list[Emitter]:
    """Draw the fixed per-device parameters.

    Uses its own RNG stream, derived from the dataset seed but independent of
    every sample's stream, so the emitter population is identical whether the
    dataset has a thousand buffers or ten million. The literal spawn key is
    arbitrary; it only has to differ from the sample indices, which occupy
    ``spawn_key=(i,)``.
    """
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(0xE117,))))

    def u(bounds: tuple[float, float]) -> float:
        return float(bounds[0]) if bounds[0] == bounds[1] else float(rng.uniform(*bounds))

    return [
        Emitter(
            emitter_id=i,
            iq_gain_db=u(prior.iq_gain_db),
            iq_phase_deg=u(prior.iq_phase_deg),
            dc_i_dbc=u(prior.dc_dbc),
            dc_q_dbc=u(prior.dc_dbc),
            pa_ibo_db=u(prior.pa_ibo_db),
            pn_linewidth_hz=u(prior.pn_linewidth_hz),
            cfo_bias_hz=u(prior.cfo_bias_hz),
        )
        for i in range(n)
    ]
