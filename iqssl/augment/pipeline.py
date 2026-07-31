"""Turn a collated batch into the views a method's ``ViewSpec`` asked for.

This is the one place that reads a :class:`~iqssl.types.ViewSpec` and acts on it,
which is what lets the training loop stay method-agnostic: the loop hands the
pipeline a batch and a spec, and never learns why a method wanted two views
rather than none.

The pipeline runs on-device, after collation, deliberately. Augmenting inside
dataloader workers would make the view distribution depend on the worker count,
and the strengths are drawn from a dedicated generator so augmentation never
perturbs the streams used for weight init or shuffling.
"""

from __future__ import annotations

import torch
from torch import Generator, Tensor

from iqssl.augment.masking import make_jepa_masks, make_mask
from iqssl.augment.policies import AugmentPolicy, get_policy
from iqssl.types import Batch, ViewSpec


class ViewPipeline:
    """Builds views and masks for one method, per its ``ViewSpec``."""

    def __init__(
        self,
        spec: ViewSpec,
        policy: AugmentPolicy | str = "standard",
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        n_tokens: int | None = None,
    ) -> None:
        self.spec = spec
        self.policy = get_policy(policy) if isinstance(policy, str) else policy
        self.n_tokens = n_tokens
        self.device = torch.device(device)
        self._g: Generator = torch.Generator(device=self.device)
        self._g.manual_seed(seed)

        if spec.needs_mask and n_tokens is None:
            raise ValueError(
                f"{spec.mask_kind!r} masking was requested but n_tokens is unknown; "
                "pass the encoder's num_patches so masks match its tokenization"
            )

    def __call__(self, batch: Batch) -> Batch:
        """Populate ``batch.views`` and ``batch.masks`` in place. Returns it.

        Mask geometry (ratio, block size) comes from the *spec*, not from
        pipeline configuration: it is part of the objective a method declared,
        and a pipeline-level knob would let an experiment change one method's
        objective while claiming to hold it fixed.
        """
        x = batch.x_raw
        batch.views = [self._view(x) for _ in range(self.spec.n_views)]

        if self.spec.needs_mask:
            assert self.spec.mask_kind is not None and self.n_tokens is not None
            if self.spec.mask_kind == "jepa":
                batch.masks = make_jepa_masks(
                    x.shape[0],
                    self.n_tokens,
                    ratio=self.spec.mask_ratio,
                    generator=self._g,
                    device=x.device,
                )
            else:
                batch.masks = {
                    "mask": make_mask(
                        x.shape[0],
                        self.n_tokens,
                        self.spec.mask_kind,
                        ratio=self.spec.mask_ratio,
                        block_size=self.spec.mask_block_size,
                        generator=self._g,
                        device=x.device,
                    )
                }
        return batch

    def _view(self, x: Tensor) -> Tensor:
        """One independently augmented view.

        Independence matters for asymmetric methods in particular: BYOL and the
        JEPA family treat view 0 and view 1 differently, so sharing a draw
        between them would make the teacher's input a deterministic function of
        the student's and collapse the prediction task.
        """
        return self.policy(x, self._g)

    def state_dict(self) -> dict:
        return {"generator": self._g.get_state()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the augmentation stream.

        Checkpoint/resume must reproduce the *same* view sequence, or a resumed
        run diverges from an uninterrupted one and the resume-equivalence test
        that guards optimizer and EMA state would silently stop being able to
        isolate them.
        """
        self._g.set_state(state["generator"])
