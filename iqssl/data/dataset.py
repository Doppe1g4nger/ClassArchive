"""Reading a built dataset, and turning items into a :class:`~iqssl.types.Batch`.

Two decisions here shape everything downstream.

**Cropping is renormalized.** A crop of a unit-power buffer is not itself unit
power, and the residual is correlated with what the crop happened to contain.
Renormalizing per crop closes that, and matters because absolute received power
is otherwise a shortcut feature perfectly correlated with SNR.

**Collate never builds views.** It assembles labels, nuisances and the raw buffer;
:mod:`iqssl.augment` produces the views. Keeping augmentation out of the
dataloader workers means the view distribution is a property of the training
config rather than of how many workers happened to be running, and it lets the
same collated batch feed methods with different ``ViewSpec``s.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from iqssl.data.build import META_NAME, SUBSETS_NAME, load_manifest
from iqssl.data.params import NUISANCE_FIELDS
from iqssl.types import Batch, ViewSpec

PRIMARY_LABELS = ("emitter", "modulation")


class IQDataset(Dataset):
    """One split of a built dataset.

    Shards are memory-mapped rather than read: at 200k buffers the array is
    several gigabytes, and every dataloader worker would otherwise hold its own
    copy.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        split_variant: str = "iid",
        primary_label: str = "emitter",
        crop_len: int | None = None,
        random_crop: bool = True,
        standardize_nuisance: bool = True,
    ) -> None:
        if primary_label not in PRIMARY_LABELS:
            raise ValueError(
                f"unknown primary_label {primary_label!r}; options: {list(PRIMARY_LABELS)}"
            )
        self.root = Path(root)
        self.split = split
        self.primary_label = primary_label
        self.random_crop = random_crop
        self.standardize_nuisance = standardize_nuisance

        self.manifest = load_manifest(self.root)
        self.dataset_hash: str = self.manifest["dataset_hash"]
        self.store_len: int = int(self.manifest["store_len"])
        self.crop_len = int(crop_len if crop_len is not None else self.manifest["crop_len"])
        if self.crop_len > self.store_len:
            raise ValueError(
                f"crop_len {self.crop_len} exceeds store_len {self.store_len}; "
                "rebuild the dataset with a longer buffer, or crop shorter"
            )

        meta = pd.read_parquet(self.root / META_NAME)
        col = f"split_{split_variant}"
        if col not in meta.columns:
            raise ValueError(
                f"split variant {split_variant!r} was not stored for this dataset; "
                f"available: {[c[6:] for c in meta.columns if c.startswith('split_')]}"
            )
        self._rows = np.flatnonzero(meta[col].to_numpy() == split)
        if len(self._rows) == 0:
            raise ValueError(f"split {split!r} of variant {split_variant!r} is empty")

        self.meta = meta
        self._split_variant = split_variant
        self._sub = meta.iloc[self._rows].reset_index(drop=True)

        self._mods = sorted(self.manifest["modulations"])
        self._mod_index = {m: i for i, m in enumerate(self._mods)}
        self.n_emitters = int(self.manifest["n_emitters"])

        self.snr = self._sub["snr_db_realized"].to_numpy().astype(np.float32)
        self._y_emitter = self._sub["emitter_id"].to_numpy().astype(np.int64)
        self._y_mod = np.array(
            [self._mod_index[m] for m in self._sub["modulation"]], dtype=np.int64
        )
        self._nuisance = self._sub[list(NUISANCE_FIELDS)].to_numpy().astype(np.float32)

        stats = self.manifest["nuisance_stats"]
        self._nz_mean = np.asarray(stats["mean"], dtype=np.float32)
        self._nz_std = np.asarray(stats["std"], dtype=np.float32)

        self._shards: dict[int, np.memmap] = {}
        self._shard_of = self._sub["shard"].to_numpy().astype(np.int64)
        self._row_of = self._sub["row_in_shard"].to_numpy().astype(np.int64)

    # -- torch Dataset ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> dict:
        raw = self._read(i)
        x = self._crop(torch.from_numpy(np.array(raw, dtype=np.float32)))

        nz = self._nuisance[i]
        if self.standardize_nuisance:
            nz = (nz - self._nz_mean) / self._nz_std

        y_em = int(self._y_emitter[i])
        y_mod = int(self._y_mod[i])
        return {
            "x": x,
            "y_emitter": y_em,
            "y_mod": y_mod,
            "y_primary": y_em if self.primary_label == "emitter" else y_mod,
            "nuisance": torch.from_numpy(np.ascontiguousarray(nz)),
            "index": int(self._rows[i]),
        }

    # -- helpers ---------------------------------------------------------------

    @property
    def num_primary_classes(self) -> int:
        return self.n_emitters if self.primary_label == "emitter" else len(self._mods)

    def primary_labels(self) -> np.ndarray:
        """All primary labels for this split, ``(N,)`` int64.

        Exposed so the training loop can build a class-balanced sampler without
        materializing every item first -- at 200k buffers that would mean reading
        the whole array off disk before the first step.
        """
        return self._y_emitter if self.primary_label == "emitter" else self._y_mod

    def _read(self, i: int) -> np.ndarray:
        s = int(self._shard_of[i])
        if s not in self._shards:
            self._shards[s] = np.load(self.root / f"shard_{s:04d}.npy", mmap_mode="r")
        return self._shards[s][int(self._row_of[i])]

    def _crop(self, x: Tensor) -> Tensor:
        """Take ``crop_len`` samples and renormalize to unit power.

        Renormalization is not cosmetic: a crop of a unit-power buffer carries a
        residual power that correlates with what it contained, which would hand
        the model a free feature.
        """
        slack = self.store_len - self.crop_len
        if slack == 0:
            start = 0
        elif self.random_crop:
            start = int(torch.randint(0, slack + 1, (1,)).item())
        else:
            start = slack // 2
        x = x[:, start : start + self.crop_len]

        power = (x**2).sum(0).mean().clamp_min(1e-12)
        return x / power.sqrt()

    def label_subset(self, fraction: float) -> list[int]:
        """Positions (into *this* split) of the shared few-label subset.

        Every method finetunes on exactly these samples. The list is stored at
        build time, not drawn here, so two methods cannot disagree about it.
        """
        with open(self.root / SUBSETS_NAME) as fh:
            subsets = json.load(fh)
        axis = "emitter_id" if self.primary_label == "emitter" else "modulation"
        try:
            rows = subsets[self._split_variant][axis][str(fraction)]
        except KeyError:
            available = sorted(subsets[self._split_variant][axis])
            raise KeyError(
                f"no stored label subset at fraction {fraction}; available: {available}"
            ) from None

        # Stored indices are global rows; translate into positions in this split.
        position = {int(r): p for p, r in enumerate(self._rows)}
        return sorted(position[r] for r in rows if r in position)


def make_collate(view_spec: ViewSpec):  # type: ignore[no-untyped-def]
    """Build a collate function for a method's :class:`~iqssl.types.ViewSpec`.

    Labels are attached **unconditionally**, even when ``needs_labels`` is False.
    The online probe that every run logs needs them, and an unsupervised method
    receiving labels in its batch cannot accidentally use them -- the objective
    simply never reads the field.
    """

    def collate(items: list[dict]) -> Batch:
        return Batch(
            x_raw=torch.stack([it["x"] for it in items]),
            views=[],  # iqssl.augment builds these; see the module docstring.
            y_emitter=torch.tensor([it["y_emitter"] for it in items], dtype=torch.long),
            y_mod=torch.tensor([it["y_mod"] for it in items], dtype=torch.long),
            y_primary=torch.tensor([it["y_primary"] for it in items], dtype=torch.long),
            nuisance=torch.stack([it["nuisance"] for it in items]),
            nuisance_names=NUISANCE_FIELDS,
            index=torch.tensor([it["index"] for it in items], dtype=torch.long),
        )

    collate.view_spec = view_spec  # type: ignore[attr-defined]
    return collate
