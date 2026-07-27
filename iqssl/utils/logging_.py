"""Run logging: CSV and TensorBoard always, W&B only if asked for.

CSV is the one that matters. TensorBoard directories rot, W&B projects get
archived, but ``metrics.csv`` sitting next to the checkpoint still parses in
five years with pandas and no credentials. The aggregator reads CSV.

W&B is imported lazily inside :class:`WandbLogger` so the package never becomes
a hard dependency — ``pip install iqssl`` must work without a W&B account.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Protocol

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_console_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Logger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class CSVLogger:
    """Append-only metrics CSV.

    Metric keys are discovered incrementally (a method may start logging a new
    scalar mid-run), so the header is rewritten when the key set grows. Rows are
    buffered and flushed on ``close`` or when the header changes, which keeps
    the file valid even if the run is killed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] = ["step"]
        self._rows: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        row: dict[str, Any] = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        new = [k for k in row if k not in self._fields]
        if new:
            self._fields.extend(sorted(new))
        self._rows.append(row)
        if len(self._rows) >= 50:
            self.flush()

    def flush(self) -> None:
        if not self._rows and self.path.exists():
            return
        # Rewriting whole-file keeps the header consistent when new keys appear.
        # Runs log a few thousand rows, so the cost is irrelevant.
        existing: list[dict[str, Any]] = []
        if self.path.exists():
            with open(self.path, newline="") as fh:
                existing = list(csv.DictReader(fh))
        all_rows = existing + self._rows
        fields = list(dict.fromkeys(["step", *(k for r in all_rows for k in r)]))
        with open(self.path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        self._fields = fields
        self._rows = []

    def close(self) -> None:
        self.flush()


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(log_dir))

    def log(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class WandbLogger:
    def __init__(self, project: str, name: str, config: dict[str, Any] | None = None) -> None:
        import wandb

        self._run = wandb.init(project=project, name=name, config=config, reinit=True)

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._run.log(metrics, step=step)

    def close(self) -> None:
        self._run.finish()


class MultiLogger:
    """Fans metrics out to several backends; a broken backend never kills a run."""

    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = loggers
        self._log = get_logger(__name__)

    def log(self, metrics: dict[str, float], step: int) -> None:
        for lg in self._loggers:
            try:
                lg.log(metrics, step)
            except Exception as exc:
                self._log.warning("logger %s failed: %s", type(lg).__name__, exc)

    def close(self) -> None:
        for lg in self._loggers:
            try:
                lg.close()
            except Exception as exc:
                self._log.warning("logger %s failed to close: %s", type(lg).__name__, exc)


def build_logger(
    run_dir: str | Path,
    *,
    csv: bool = True,
    tensorboard: bool = True,
    wandb: bool = False,
    wandb_project: str = "iqssl",
    run_name: str = "run",
    config: dict[str, Any] | None = None,
) -> MultiLogger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    loggers: list[Logger] = []
    if csv:
        loggers.append(CSVLogger(run_dir / "metrics.csv"))
    if tensorboard:
        loggers.append(TensorBoardLogger(run_dir / "tb"))
    if wandb:
        loggers.append(WandbLogger(wandb_project, run_name, config))
    return MultiLogger(loggers)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)
