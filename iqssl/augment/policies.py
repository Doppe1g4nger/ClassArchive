"""Named augmentation policies — the benchmark's second experimental axis.

A policy is an ordered list of ops with per-op application probabilities. Five
ship, forming a ladder plus one control:

============================  ===========================================
``none``                      identity; masked methods under ``augment=none``
``light``                     phase and small timing only
``standard``                  channel invariance -- **the default**
``heavy``                     the same axes, pushed harder, plus occlusion
``hardware_invariant``        positive control: destroys the fingerprint
============================  ===========================================

``standard`` deliberately touches nothing that carries emitter identity. That is
the central design commitment: a contrastive objective becomes invariant to
whatever its augmentations vary, so augmenting IQ imbalance away does not make
the task harder, it deletes the label.

``hardware_invariant`` exists to prove that claim rather than assert it. It
randomizes exactly the impairments the emitter label is made of, and emitter
accuracy under it is *expected* to collapse toward chance while modulation
accuracy survives. If it does not collapse, then the fingerprint is not where the
generator says it is, and every emitter result in the thesis is suspect. A
positive control that cannot fail is not a control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Generator, Tensor

from iqssl.augment import ops


@dataclass(frozen=True)
class OpSpec:
    """One op in a policy: what to call, how often, and with what arguments."""

    name: str
    p: float = 1.0
    kwargs: dict = field(default_factory=dict)

    def __call__(self, x: Tensor, g: Generator | None) -> Tensor:
        fn = ops.AUG_OPS.get(self.name)  # type: ignore[attr-defined]
        return fn(x, g, **self.kwargs)  # type: ignore[operator]


@dataclass(frozen=True)
class AugmentPolicy:
    """An ordered op chain applied independently to each view."""

    name: str
    specs: tuple[OpSpec, ...] = ()
    renormalize: bool = True
    touches_fingerprint: bool = False
    """True for policies that perturb emitter-identifying impairments.

    Surfaced as a flag rather than left implicit so the analysis stage can refuse
    to report an emitter result from a fingerprint-destroying policy as though it
    were a normal one.
    """

    def __call__(self, x: Tensor, g: Generator | None = None) -> Tensor:
        out = x
        for spec in self.specs:
            if spec.p >= 1.0:
                out = spec(out, g)
                continue
            # Per-sample application. Applying per *batch* would correlate the
            # augmentation choice across a batch, and for a contrastive loss the
            # batch is the unit of comparison -- every negative would share the
            # anchor's augmentation and the objective would quietly change.
            applied = spec(out, g)
            keep = (torch.rand(x.shape[0], generator=g, device=x.device) < spec.p).view(-1, 1, 1)
            out = torch.where(keep, applied, out)
        return ops.renormalize(out) if self.renormalize else out


def _op(name: str, p: float = 1.0, **kwargs) -> OpSpec:  # type: ignore[no-untyped-def]
    return OpSpec(name=name, p=p, kwargs=kwargs)


POLICIES: dict[str, AugmentPolicy] = {
    "none": AugmentPolicy(name="none", specs=(), renormalize=False),
    "light": AugmentPolicy(
        name="light",
        specs=(
            _op("phase_rotate"),
            _op("time_shift", p=0.5, max_samples=2.0),
        ),
    ),
    # `standard` is calibrated, not guessed. Holding the signal and channel fixed
    # and varying only the emitter, the largest standardized separation between
    # two contrasting devices across the classical features moves like this:
    #
    #     no augmentation ................................. d = 13.3
    #     phase + CFO + timing + clock drift .............. d = 13.3   (no cost)
    #     + renoise, p=0.5, SNR 10-30 dB .................. d =  3.4
    #     + multipath, p=0.5, 2 taps, 1.0 symbol spread ... d =  1.5
    #     + multipath, p=0.3, 2 taps, 0.5 symbol spread ... d =  2.4
    #
    # Phase and timing nuisances are free -- they cost the fingerprint nothing,
    # which is why `light` consists of exactly those. Noise and fading are not:
    # both genuinely obscure hardware signatures, and that is physics rather than
    # a defect. But an augmentation strong enough to erase the emitter label
    # would floor every method at chance and the method x policy interaction the
    # thesis reports would have no signal left to resolve. Multipath is therefore
    # kept mild here and pushed hard in `heavy`, where the floor effect is the
    # point rather than an accident.
    "standard": AugmentPolicy(
        name="standard",
        specs=(
            _op("phase_rotate"),
            _op("cfo_jitter", p=0.8, max_norm=5e-4),
            _op("time_shift", p=0.8, max_samples=4.0),
            _op("sample_rate_offset", p=0.5, max_ppm=30.0),
            _op("multipath", p=0.3, n_taps=2, delay_spread_symbols=0.5),
            _op("renoise", p=0.5, snr_db=(15.0, 30.0)),
        ),
    ),
    "heavy": AugmentPolicy(
        name="heavy",
        specs=(
            _op("phase_rotate"),
            _op("cfo_jitter", p=1.0, max_norm=2e-3),
            _op("time_shift", p=1.0, max_samples=8.0),
            _op("sample_rate_offset", p=0.8, max_ppm=80.0),
            _op("multipath", p=0.8, n_taps=3, delay_spread_symbols=2.0),
            _op("renoise", p=0.8, snr_db=(0.0, 20.0)),
            _op("time_mask", p=0.5, max_frac=0.15),
        ),
    ),
    "hardware_invariant": AugmentPolicy(
        name="hardware_invariant",
        # The channel ops from `standard`, plus the three that randomize the
        # fingerprint itself. Applied with p=1.0: a control that only sometimes
        # fires would only sometimes destroy the label, and the result would be
        # ambiguous rather than decisive.
        specs=(
            _op("phase_rotate"),
            _op("cfo_jitter", p=0.8, max_norm=5e-4),
            _op("time_shift", p=0.8, max_samples=4.0),
            _op("sample_rate_offset", p=0.5, max_ppm=30.0),
            _op("multipath", p=0.3, n_taps=2, delay_spread_symbols=0.5),
            _op("renoise", p=0.5, snr_db=(15.0, 30.0)),
            # Strengths deliberately exceed the emitter prior in
            # iqssl.data.params (1.2 dB / 9 deg, -30..-19 dBc, 3-11 dB IBO).
            # Augmentation is a forward operation: it cannot un-bake the PA
            # compression already in the waveform, only add more on top. To
            # dominate the emitter's own signature rather than merely perturb it,
            # the jitter has to be the larger of the two.
            _op("iq_imbalance_jitter", p=1.0, max_gain_db=3.0, max_phase_deg=20.0),
            _op("dc_jitter", p=1.0, dbc=(-26.0, -14.0)),
            _op("pa_jitter", p=1.0, ibo_db=(2.0, 8.0)),
        ),
        touches_fingerprint=True,
    ),
}


def get_policy(name: str) -> AugmentPolicy:
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown augment policy {name!r}; options: {sorted(POLICIES)}") from None
