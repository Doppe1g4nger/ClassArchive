"""The shared training loop.

One loop trains every objective, and it never branches on method type. That is
the mechanism by which the benchmark is fair rather than merely careful: if the
loop had to special-case SimCLR's two views or MAE's masking, each special case
would be an opportunity for one method to receive a scheduling, batching or
normalization advantage the others never got, and no amount of matched
hyperparameters afterwards would recover the comparison.

Everything method-specific reaches the loop through the four
:class:`~iqssl.methods.base.Method` hooks and the
:class:`~iqssl.types.ViewSpec`.
"""

from iqssl.train.loop import TrainState, train
from iqssl.train.optim import build_optimizer
from iqssl.train.schedules import cosine_with_warmup

__all__ = ["TrainState", "build_optimizer", "cosine_with_warmup", "train"]
