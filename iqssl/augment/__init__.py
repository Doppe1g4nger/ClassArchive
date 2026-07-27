"""Train-time view transforms: what counts as a positive pair.

This package wraps the *same* :mod:`iqssl.dsp` primitives that
:mod:`iqssl.data` composes into the generative distribution, but to a different
end. Sharing the primitives is deliberate; coupling the two pipelines is not. If
augmentation reached into the generator, a change to the view distribution could
silently shift the data distribution, which is exactly the confound this
benchmark exists to control.

**The augmentation policy is a first-class experimental axis, not a fixed prior.**
Contrastive methods become invariant to whatever you augment with -- and IQ
imbalance, DC offset and PA nonlinearity *are* the emitter label. Augmenting them
away does not regularize the task, it deletes it. So ``standard`` is
channel-invariance only, ``hardware_invariant`` ships as a positive control that
should visibly destroy emitter accuracy, and results report the method x policy
interaction rather than comparing one method against another method plus a
hand-designed prior.
"""

from iqssl.augment.masking import make_mask
from iqssl.augment.pipeline import ViewPipeline
from iqssl.augment.policies import POLICIES, AugmentPolicy, get_policy

__all__ = ["POLICIES", "AugmentPolicy", "ViewPipeline", "get_policy", "make_mask"]
