"""Seeding and RNG discipline.

Two separate concerns live here:

*Training* reproducibility — one global seed, plus per-worker dataloader seeds
derived from it, plus an optional strict-determinism mode used by the tests that
assert two runs produce identical loss traces.

*Generation* reproducibility — the dataset generator does **not** use a global
RNG. It spawns an independent :class:`numpy.random.SeedSequence` per sample, so
generation is embarrassingly parallel, order-independent and byte-reproducible:
sample 12345 is the same signal whether you generate 100 samples or 10 million,
and whether you generate them in one process or thirty-two.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    ``deterministic=True`` additionally forbids nondeterministic CUDA kernels.
    It is meaningfully slower, so it is off by default and on in the tests that
    compare loss traces.
    """
    random.seed(seed)
    # NPY002: seeding the *legacy* global RNG is the point. scikit-learn and
    # other third-party code still draws from it, and an unseeded global stream
    # is exactly the leak this function exists to close. Our own generation code
    # uses SeedSequence-spawned Generators (see spawn_streams).
    np.random.seed(seed % (2**32))  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct but reproducible RNG stream.

    Torch already seeds ``torch.initial_seed()`` per worker; the bug this guards
    against is numpy and python ``random`` staying identical across workers,
    which silently duplicates every random draw made outside torch.
    """
    base = torch.initial_seed() % (2**32)
    np.random.seed((base + worker_id) % (2**32))  # noqa: NPY002 - see seed_everything
    random.seed(base + worker_id)


def spawn_streams(base_seed: int, n: int) -> list[np.random.Generator]:
    """``n`` independent PCG64 generators derived from ``base_seed``.

    Uses :class:`numpy.random.SeedSequence` spawning, which guarantees the
    streams are statistically independent — unlike ``seed + i``, which for many
    bit generators produces correlated sequences.
    """
    return [
        np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(base_seed).spawn(n)
    ]


def spawn_keys(base_seed: int, n: int) -> list[int]:
    """Stable integer keys, one per sample, derived the same way.

    Stored in the dataset metadata so any intermediate quantity (the clean
    signal, the channel taps) can be *regenerated* on demand rather than stored.
    """
    return [
        int(s.generate_state(1, dtype=np.uint32)[0])
        for s in np.random.SeedSequence(base_seed).spawn(n)
    ]


def generator_from_key(key: int) -> np.random.Generator:
    """Rebuild the generator for a stored spawn key."""
    return np.random.Generator(np.random.PCG64(key))


def sample_stream(base_seed: int, index: int) -> np.random.Generator:
    """The RNG for one dataset sample, in O(1) and without materializing others.

    ``SeedSequence(entropy=s, spawn_key=(i,))`` is by construction identical to
    ``SeedSequence(s).spawn(n)[i]`` for every ``n > i``, so this is both cheap
    and prefix-stable: sample ``i`` is the same signal whether the dataset has a
    thousand samples or ten million, generated in one process or thirty-two.
    """
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(base_seed, spawn_key=(index,)))
    )


def torch_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """A torch generator for augmentation, kept separate from the global RNG.

    Augmentation draws must not perturb model-init or shuffling streams, or a
    change to the augmentation policy would silently change weight init too.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@contextlib.contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    """Temporarily seed torch/numpy/random, restoring the prior state on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()  # noqa: NPY002 - saving legacy global state
    torch_state = torch.get_rng_state()
    try:
        seed_everything(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)  # noqa: NPY002 - restoring legacy global state
        torch.set_rng_state(torch_state)
