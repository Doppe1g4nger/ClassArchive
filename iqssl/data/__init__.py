"""The generative data distribution, and the gate that says whether it is usable.

This package composes :mod:`iqssl.dsp` primitives into signals and records the
ground-truth value of every nuisance parameter it applied. It is deliberately
separate from :mod:`iqssl.augment`, which wraps the *same* primitives as
train-time view transforms: sharing the primitives is the point, coupling the two
pipelines is not. If they were one module, a change to the augmentation policy
could silently shift the data distribution, which is precisely the confound this
benchmark exists to control.

Three label axes come out of generation — modulation class, emitter id, and a
continuous nuisance vector — and emitter parameters are drawn statistically
independent of channel parameters. That independence is load-bearing: if carrier
frequency offset partly encoded emitter identity, then "did the representation
discard CFO?" would have no interpretable answer, because discarding it would
help one task while destroying the other.
"""

from iqssl.data.params import NUISANCE_FIELDS, PRESETS, GeneratorConfig, get_preset

__all__ = ["NUISANCE_FIELDS", "PRESETS", "GeneratorConfig", "get_preset"]
