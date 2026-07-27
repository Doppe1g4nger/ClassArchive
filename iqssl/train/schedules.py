"""Learning-rate schedule: 10% warmup, then cosine decay.

Held identical across every method, and applied per parameter *group* so that a
method can ask for different treatment of a head without the loop knowing which
method it is serving. Two group flags are honoured, and both come from published
recipes rather than from taste:

``lr_scale``
    Multiplies the group's peak LR. BYOL's predictor runs at 10x.
``fix_lr``
    Holds the group at its base LR for the whole run, skipping warmup and decay.
    SimSiam's predictor requires this -- decaying it measurably collapses the
    representation, and it is one of the few genuinely load-bearing details in
    that paper.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer


def cosine_with_warmup(step: int, total_steps: int, warmup_frac: float = 0.1) -> float:
    """LR multiplier in ``[0, 1]``: linear warmup, then cosine to zero."""
    if total_steps <= 1:
        return 1.0
    warmup = max(1, int(warmup_frac * total_steps))
    if step < warmup:
        # step+1 so the very first step is not exactly zero, which would waste it.
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def apply_lr(optimizer: Optimizer, step: int, total_steps: int, warmup_frac: float = 0.1) -> float:
    """Set every group's LR for this step. Returns the representative LR logged.

    The reported value is the first non-``fix_lr`` group's, since that is the one
    the schedule actually describes.
    """
    scale = cosine_with_warmup(step, total_steps, warmup_frac)
    reported = None
    for group in optimizer.param_groups:
        base = group.setdefault("base_lr", group["lr"])
        if group.get("fix_lr", False):
            group["lr"] = base
            continue
        group["lr"] = base * group.get("lr_scale", 1.0) * scale
        if reported is None:
            reported = group["lr"]
    return float(reported if reported is not None else optimizer.param_groups[0]["lr"])


def scale_base_lr(base_lr: float, batch_size: int, reference: int = 256) -> float:
    """The linear scaling rule, ``lr = base_lr * B / 256``.

    Applied identically to every method so that batch size stays a held-constant
    rather than an accidental per-method advantage.
    """
    return base_lr * batch_size / reference
