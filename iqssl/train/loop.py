"""The shared training loop.

Read the body and note what is *absent*: there is no ``if method == ...``
anywhere. Every objective reaches the loop through four hooks and a
:class:`~iqssl.types.ViewSpec`, and that is the mechanism by which the benchmark
is fair rather than merely careful. A single special case -- SimCLR getting its
views built differently, MAE's masking handled inline -- would be a place where
one method could acquire a scheduling or batching advantage no other method got,
and no amount of matched hyperparameters afterwards would recover the comparison.

The loop's own responsibilities are exactly: build the batch the spec asks for,
apply the shared schedule, step, let the method update its EMA, and record what
was spent.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from iqssl.augment.pipeline import ViewPipeline
from iqssl.data.dataset import IQDataset, make_collate
from iqssl.methods.base import Method
from iqssl.train.optim import build_optimizer
from iqssl.train.schedules import apply_lr, scale_base_lr
from iqssl.utils import env
from iqssl.utils.logging_ import build_logger, get_logger, write_json
from iqssl.utils.meters import ComputeMeter, MeterDict
from iqssl.utils.seed import seed_everything, worker_init_fn

log = get_logger(__name__)


@dataclass
class TrainConfig:
    """Everything the loop needs. Held constant across methods except where the
    fairness contract explicitly allows variation (optimizer, lr, wd)."""

    epochs: int = 10
    batch_size: int = 256
    base_lr: float = 1e-3
    weight_decay: float = 1e-6
    optimizer: str = "adamw"
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    num_workers: int = 0
    device: str = "cpu"
    augment: str = "standard"
    log_every: int = 20
    probe_every: int = 0
    """Steps between online-probe updates. 0 disables it."""

    max_steps: int | None = None
    """Cap for smoke runs. None means epochs decide."""


@dataclass
class TrainState:
    """What a finished run produced."""

    step: int = 0
    epoch: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    compute: dict[str, float] = field(default_factory=dict)
    final_loss: float = float("nan")


def build_loader(
    dataset: IQDataset, method_cls: type[Method], cfg: TrainConfig, cfg_obj: Any = None
) -> DataLoader:
    """Dataloader whose collate and sampler follow the method's ``ViewSpec``.

    The class-balanced sampler is switched on by ``needs_labels`` rather than by
    method name. SupCon is the reason: with many classes and a modest batch, a
    uniform sampler leaves most anchors with no positive in the batch at all, and
    the loss silently degenerates toward NT-Xent -- it does not error, it just
    stops being the objective you think you are running.
    """
    spec = method_cls.view_spec(cfg_obj)
    sampler = None
    shuffle = True
    if spec.needs_labels:
        labels = dataset.primary_labels()
        counts = torch.bincount(torch.from_numpy(labels))
        weights = (1.0 / counts.clamp_min(1).float())[torch.from_numpy(labels)]
        sampler = WeightedRandomSampler(weights.tolist(), len(dataset), replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=make_collate(spec),
        drop_last=True,
        worker_init_fn=worker_init_fn if cfg.num_workers else None,
        persistent_workers=cfg.num_workers > 0,
    )


def train(
    method: Method,
    dataset: IQDataset,
    cfg: TrainConfig,
    *,
    out_dir: str | Path | None = None,
    method_cfg: Any = None,
) -> TrainState:
    """Pretrain one method. Returns the run state; writes logs to ``out_dir``."""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    method = method.to(device)

    spec = type(method).view_spec(method_cfg)
    loader = build_loader(dataset, type(method), cfg, method_cfg)

    n_tokens = getattr(method.encoder, "num_patches", None)
    pipeline = ViewPipeline(
        spec,
        cfg.augment,
        # Offset from the run seed so the augmentation stream is reproducible
        # but distinct from the streams driving init and shuffling.
        seed=cfg.seed + 1000,
        device=device,
        n_tokens=n_tokens,
    )

    steps_per_epoch = len(loader)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs

    lr = scale_base_lr(cfg.base_lr, cfg.batch_size)
    optimizer = build_optimizer(
        method.param_groups(lr, cfg.weight_decay),
        cfg.optimizer,
        lr=lr,
        weight_decay=cfg.weight_decay,
    )

    out = Path(out_dir) if out_dir else None
    logger = build_logger(str(out)) if out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "run_meta.json", asdict(env.capture()))
        write_json(
            out / "config.json",
            {
                "train": asdict(cfg),
                "method": type(method).__name__,
                "view_spec": asdict(spec),
                "dataset_hash": dataset.dataset_hash,
                "total_steps": total_steps,
                "lr_scaled": lr,
            },
        )

    compute = ComputeMeter()
    meters = MeterDict()
    state = TrainState()
    probe = _OnlineProbe(method.embed_dim, dataset.num_primary_classes, device)

    log.info(
        "training %s for %d steps (%d epochs x %d) on %d samples",
        type(method).__name__,
        total_steps,
        cfg.epochs,
        steps_per_epoch,
        len(dataset),
    )

    method.train()
    t0 = time.perf_counter()
    done = False
    for epoch in range(cfg.epochs):
        if done:
            break
        state.epoch = epoch
        for batch in loader:
            if state.step >= total_steps:
                done = True
                break

            batch = batch.to(device)
            batch = pipeline(batch)

            current_lr = apply_lr(optimizer, state.step, total_steps, cfg.warmup_frac)
            optimizer.zero_grad(set_to_none=True)

            out_ = method(batch, state.step, total_steps)
            out_.loss.backward()

            grad_norm = float(
                nn.utils.clip_grad_norm_(method.parameters(), cfg.grad_clip)
                if cfg.grad_clip > 0
                else 0.0
            )
            optimizer.step()

            # After the optimizer step, which is what the published EMA
            # algorithms specify -- updating before it makes the teacher track a
            # student that no longer exists.
            method.on_step_end(state.step, total_steps)

            # The method declares its own spend (see encoder_passes_per_step);
            # the loop just books it. Measuring here instead would need the loop
            # to know which forwards were teacher passes, i.e. to branch.
            student_passes, teacher_passes = method.encoder_passes_per_step()
            tokens = n_tokens or batch.seq_len
            compute.add_forward(batch.batch_size, tokens, weight=student_passes)
            if teacher_passes > 0:
                compute.add_forward(batch.batch_size, tokens, teacher=True, weight=teacher_passes)
            compute.add_step(batch.batch_size)
            compute.update_peak_memory()

            logs = dict(out_.logs)
            logs["lr"] = current_lr
            logs["grad_norm"] = grad_norm
            if cfg.probe_every and state.step % cfg.probe_every == 0:
                logs.update(probe.update(method, batch))
            meters.update(logs)

            if state.step % cfg.log_every == 0:
                row = {**meters.averages(), **compute.snapshot(), "epoch": float(epoch)}
                state.history.append({"step": float(state.step), **row})
                if logger:
                    logger.log(row, step=state.step)
                log.info(
                    "step %d/%d  loss %.4f  lr %.2e",
                    state.step,
                    total_steps,
                    logs.get("loss", float("nan")),
                    current_lr,
                )
                meters.reset()

            state.final_loss = float(out_.loss.detach())
            state.step += 1

    state.compute = compute.snapshot()
    state.compute["compute/wall_clock_s"] = time.perf_counter() - t0

    if out:
        torch.save(
            {
                "method": method.state_dict(),
                "optimizer": optimizer.state_dict(),
                "pipeline": pipeline.state_dict(),
                "step": state.step,
            },
            out / "checkpoint.pt",
        )
        write_json(out / "summary.json", {"final_loss": state.final_loss, **state.compute})
    if logger:
        logger.close()
    return state


class _OnlineProbe:
    """A linear classifier on detached features, logged during training.

    Detached deliberately: it is a *diagnostic*, and letting its gradient reach
    the encoder would turn every self-supervised run into a semi-supervised one.
    Its own optimizer is separate for the same reason.
    """

    def __init__(self, dim: int, n_classes: int, device: torch.device) -> None:
        self.head = nn.Linear(dim, n_classes).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=1e-3)

    def update(self, method: Method, batch: Any) -> dict[str, float]:
        if batch.y_primary is None:
            return {}
        with torch.no_grad():
            # Both encoders return EncoderOut, and ResNet1D's `cls` field is its
            # global-average pool, so mean pooling is defined for both. No
            # isinstance branch: an encoder that did not satisfy the contract
            # should fail loudly here rather than be silently accommodated.
            h = method.encoder(batch.x_raw).pooled("mean")
        logits = self.head(h.detach())
        loss = nn.functional.cross_entropy(logits, batch.y_primary)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {
            "probe_loss": float(loss.detach()),
            "probe_acc": float((logits.argmax(-1) == batch.y_primary).float().mean()),
        }
