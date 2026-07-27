"""IQSSL — a controlled benchmark for self-supervised learning on IQ signal buffers.

The package is organised so that the *only* thing that differs between two
comparable experiments is the SSL objective:

``iqssl.dsp``
    Pure, batched, device-agnostic signal-processing primitives on ``complex64``.
``iqssl.data``
    Composes ``dsp`` primitives into the *generative* data distribution, and
    records the ground-truth value of every nuisance parameter it applied.
``iqssl.augment``
    Wraps the same ``dsp`` primitives as *train-time* view transforms.
``iqssl.models``
    Encoders (shared and held fixed across methods) and heads.
``iqssl.methods``
    One module per SSL objective, all behind a single ``Method`` contract.
``iqssl.train``
    The shared training loop; it never branches on method type.
``iqssl.eval``
    One evaluation protocol, applied identically to every method.
``iqssl.analysis``
    Run collection, fairness-invariant enforcement, thesis tables and figures.
"""

__version__ = "0.1.0"
