"""Run provenance capture.

Every run writes a ``run_meta.json``. When a thesis figure looks wrong a year
later, this file is the difference between "which commit produced this?" and a
shrug. The ``git_dirty`` flag matters most: a run from an uncommitted working
tree is not reproducible and the aggregator should say so.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class RunMeta:
    git_sha: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False
    python: str = ""
    platform: str = ""
    torch_version: str = ""
    cuda_available: bool = False
    device_name: str = "cpu"
    world_size: int = 1

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture(**extra: Any) -> RunMeta:
    """Snapshot the environment. Never raises — provenance must not break a run."""
    import torch

    status = _git("status", "--porcelain")
    cuda = torch.cuda.is_available()
    return RunMeta(
        git_sha=_git("rev-parse", "HEAD"),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_dirty=bool(status),
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=cuda,
        device_name=torch.cuda.get_device_name(0) if cuda else "cpu",
        extra=dict(extra),
    )


def pip_freeze() -> list[str]:
    """Resolved dependency versions, for the record. Empty list on failure."""
    for cmd in (["uv", "pip", "freeze"], [sys.executable, "-m", "pip", "freeze"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            if out.returncode == 0:
                return out.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            continue
    return []
