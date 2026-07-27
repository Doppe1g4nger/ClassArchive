"""Signal-processing primitives: pure, batched, device-agnostic, ``complex64``.

Everything here operates on ``(B, L)`` complex tensors and has no notion of
datasets, emitters or training. That separation is deliberate: the same
primitives are composed two different ways.

``iqssl.data.generator``
    composes them into the *generative* distribution — what the signals are.
``iqssl.augment.ops``
    wraps them as *train-time* view transforms — what a positive pair is.

Coupling those two would mean a change to the augmentation policy could
silently change the data distribution, which is precisely the confound this
benchmark exists to control.
"""

from iqssl.dsp.convert import complex_to_ri, ri_to_complex
from iqssl.dsp.power import add_awgn, measure_snr_db, normalize_rms, signal_power

__all__ = [
    "add_awgn",
    "complex_to_ri",
    "measure_snr_db",
    "normalize_rms",
    "ri_to_complex",
    "signal_power",
]
