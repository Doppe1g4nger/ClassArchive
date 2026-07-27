"""Optimizers, including LARS.

The optimizer *family* is one of the few things the fairness contract lets vary
between methods, and that is a deliberate concession rather than an oversight.
SimCLR and Barlow Twins were published with LARS at large batch, BYOL with LARS,
SimSiam with SGD, and the masked methods with AdamW. Forcing one family on all of
them would handicap several against their published behaviour and the comparison
would measure "how well does this objective tolerate AdamW" instead of "how good
is this objective". What is held constant is the *budget*: the tuning trials, the
schedule, the epochs, the seeds.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling, over SGD with momentum.

    Each parameter tensor gets its own trust-ratio ``||w|| / ||grad + wd*w||``,
    which is what makes large-batch contrastive training stable.

    Groups flagged ``lars_exclude`` skip the adaptation entirely and fall back to
    plain SGD. That flag is set on every 1-D parameter -- biases, norm gains,
    learned tokens -- because the trust ratio is meaningless for them and applying
    it measurably hurts. ``methods.base.default_param_groups`` already emits the
    flag, so no method has to remember.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            wd = group["weight_decay"]
            momentum = group["momentum"]
            excluded = group.get("lars_exclude", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p: Tensor = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)

                if not excluded:
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(d_p)
                    # A zero-initialized tensor has no scale to trust, so the
                    # ratio is undefined rather than infinite: fall back to 1.
                    trust = torch.where(
                        (w_norm > 0) & (g_norm > 0),
                        group["trust_coefficient"] * w_norm / (g_norm + group["eps"]),
                        torch.ones_like(w_norm),
                    )
                    d_p = d_p.mul(trust)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(d_p)
                p.add_(buf, alpha=-group["lr"])

        return loss


OPTIMIZERS = ("lars", "sgd", "adamw")


def build_optimizer(
    param_groups: list[dict[str, Any]],
    name: str = "adamw",
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    momentum: float = 0.9,
    betas: tuple[float, float] = (0.9, 0.999),
) -> Optimizer:
    """Construct an optimizer over method-supplied parameter groups.

    ``weight_decay`` is *not* passed at the optimizer level: the groups already
    carry per-group values, and the near-universal SSL convention of exempting
    1-D parameters is encoded there. Passing a global value would silently
    override it and re-introduce decay on norm gains and biases.
    """
    if name == "lars":
        return LARS(param_groups, lr=lr, momentum=momentum)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    raise ValueError(f"unknown optimizer {name!r}; options: {list(OPTIMIZERS)}")
