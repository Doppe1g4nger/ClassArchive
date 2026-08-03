"""The difficulty-calibration gate.

    iqssl-difficulty-report --data data/synth_v1
    iqssl-difficulty-report --data data/synth_v1 --label modulation

Runs three reference estimators against a built dataset and prints whether the
task sits in a band where SSL methods can actually be told apart. Exits nonzero
when it does not, so CI and the build pipeline can treat it as a gate rather
than as advice.

Why this runs *before* any SSL method exists: if a handful of closed-form RF
statistics already identify the emitter, then every method scores ~99%, the
ranking is noise, and the entire comparison measures nothing. The opposite
failure -- a task no supervised model can learn -- is equally fatal and equally
invisible from inside a single training run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from iqssl.data.baselines import (
    SUB_THRESHOLD_SNR_DB,
    BaselineResult,
    bands_for,
    run_classical_baseline,
    run_raw_linear_baseline,
    run_supervised_cnn_baseline,
)
from iqssl.data.build import load_manifest
from iqssl.data.dataset import IQDataset
from iqssl.utils.logging_ import get_logger, setup_console_logging, write_json

log = get_logger(__name__)


def _stack(ds: IQDataset, idx: np.ndarray) -> torch.Tensor:
    return torch.stack([ds[int(i)]["x"] for i in idx])


def _subsample(n: int, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n)[:cap] if n > cap else np.arange(n)


def run_report(
    root: str | Path,
    *,
    label: str = "emitter",
    n_train: int = 12000,
    n_test: int = 3000,
    epochs: int = 60,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    train = IQDataset(root, "train", primary_label=label, random_crop=False)  # type: ignore[arg-type]
    test = IQDataset(root, "test", primary_label=label, random_crop=False)  # type: ignore[arg-type]

    tr_idx = _subsample(len(train), n_train, seed)
    te_idx = _subsample(len(test), n_test, seed + 1)

    log.info("loading %d train / %d test buffers", len(tr_idx), len(te_idx))
    xtr, xte = _stack(train, tr_idx), _stack(test, te_idx)
    ytr = np.array([train[int(i)]["y_primary"] for i in tr_idx])
    yte = np.array([test[int(i)]["y_primary"] for i in te_idx])
    snr_te = test.snr[te_idx]

    results: list[BaselineResult] = []
    log.info("classical features ...")
    results.append(run_classical_baseline(xtr, ytr, xte, yte))
    log.info("raw-IQ linear probe ...")
    results.append(run_raw_linear_baseline(xtr, ytr, xte, yte))
    log.info("supervised CNN oracle (%d epochs) ...", epochs)
    results.append(
        run_supervised_cnn_baseline(
            xtr, ytr, xte, yte, snr_te, epochs=epochs, device=device, seed=seed
        )
    )

    snr_all = np.concatenate([train.snr, test.snr])
    preset = str(load_manifest(root).get("config", {}).get("difficulty") or "")
    checks = _evaluate_bands(results, (float(snr_all.min()), float(snr_all.max())), preset)
    return {
        "dataset": str(root),
        "dataset_hash": train.dataset_hash,
        "label": label,
        "n_classes": train.num_primary_classes,
        "chance": 1.0 / train.num_primary_classes,
        "results": [r.__dict__ for r in results],
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
    }


def _evaluate_bands(
    results: list[BaselineResult], snr_range: tuple[float, float], preset: str = ""
) -> list[dict]:
    by_name = {r.name: r for r in results}
    checks = []

    # Per-rung, not global: `medium` and `hard` exist to be harder, so grading
    # them against `easy`'s band failed them by construction. See PRESET_BANDS.
    for key, band in bands_for(preset).items():
        if key == "supervised_cnn_high_snr":
            value = by_name["supervised_cnn"].accuracy_high_snr
        elif key == "supervised_cnn_sub_threshold":
            value = by_name["supervised_cnn"].accuracy_sub_threshold
        else:
            value = by_name[key].accuracy

        # A check with no data to evaluate is *not applicable*, not failed.
        # `easy` spans 15-30 dB and so has no buffers below the fingerprinting
        # floor at all; reporting that as a failure would tell the user to fix
        # a preset behaving exactly as designed.
        applicable = not (
            key == "supervised_cnn_sub_threshold" and not _has_sub_threshold(snr_range)
        )
        if not applicable:
            checks.append(
                {
                    "check": key,
                    "value": float("nan"),
                    "band": list(band),
                    "ok": True,
                    "applicable": False,
                    "advice": (
                        f"not applicable: preset SNR range {snr_range} is entirely "
                        f"at or above the {SUB_THRESHOLD_SNR_DB:g} dB fingerprinting floor"
                    ),
                }
            )
            continue

        ok = bool(band[0] <= value <= band[1]) if not np.isnan(value) else False
        checks.append(
            {
                "check": key,
                "value": float(value),
                "band": list(band),
                "ok": ok,
                "applicable": True,
                "advice": _advice(key, value, band, by_name["supervised_cnn"].accuracy_train),
            }
        )
    return checks


def _has_sub_threshold(snr_range: tuple[float, float]) -> bool:
    """Does the preset put any buffers below the fingerprinting floor?"""
    return snr_range[0] < SUB_THRESHOLD_SNR_DB


def _advice(
    key: str, value: float, band: tuple[float, float], train_acc: float = float("nan")
) -> str:
    if np.isnan(value):
        return "not measurable -- is that SNR band populated?"
    if band[0] <= value <= band[1]:
        return ""
    if key in ("classical", "raw_linear") and value > band[1]:
        return (
            "task is too easy: the fingerprint is trivially extractable, so every "
            "SSL method will saturate. Narrow the emitter impairment spreads "
            "(EmitterPrior in iqssl/data/params.py) or widen the channel nuisances."
        )
    if key == "supervised_cnn_sub_threshold" and value > band[1]:
        # The band is an upper bound on purpose. Scoring *well* below the
        # fingerprinting floor is not good news -- it means emitter identity
        # survived a channel that should have destroyed it, so the generator
        # is leaking the label through a path the physics does not allow.
        return (
            f"sub-threshold accuracy is {value:.3f}, above the {band[1]:.2f} ceiling. "
            "Below the fingerprinting floor the emitter label should be gone, so this "
            "says the generator is leaking identity through something other than the "
            "hardware impairments -- check that emitter and channel parameters are "
            "still statistically independent, and that no per-emitter quantity "
            "survives the noise (a constant DC offset is the usual culprit)."
        )
    if key.startswith("supervised_cnn") and value < band[0]:
        # "The oracle scored too low" has two opposite causes, and the fixes
        # point in opposite directions, so the advice has to know which it is.
        # Train accuracy is what separates them.
        base = (
            "task is too hard: even a supervised oracle cannot learn it, so method "
            "rankings would be noise. "
        )
        if np.isnan(train_acc):
            return base + "Run with train accuracy reported to tell memorization from underfitting."
        if train_acc > 0.95:
            return (
                base + f"The oracle memorized the training set (train {train_acc:.3f} vs "
                f"test {value:.3f}), so this is DATA-limited, not signal-limited. "
                "Raise --n-train (and --n-samples, if the train split cannot supply "
                "it) before touching any prior: on `easy`, 12k training buffers gave "
                "0.785 and 48k gave 0.891. Strengthening impairments here would make "
                "the task easier, not harder."
            )
        return (
            base + f"Train accuracy is only {train_acc:.3f}, so the oracle is underfitting "
            "rather than memorizing and more data will not help. The signal is too "
            "weak to resolve in this buffer: lengthen it (crop_len 1024 -> 4096), "
            "which is preferred over inflating impairments because that just makes "
            "the task trivially easy again."
        )
    if key.startswith("supervised_cnn") and value > band[1]:
        return "ceiling too high: methods will bunch near saturation. Narrow the emitter spreads."
    return "outside target band"


def _print_report(report: dict) -> None:
    print()
    print(f"Difficulty report -- {report['dataset']}")
    print(f"  hash    {report['dataset_hash']}")
    print(
        f"  label   {report['label']}  ({report['n_classes']} classes, "
        f"chance {report['chance']:.3f})"
    )
    print()
    print(f"  {'estimator':<24} {'accuracy':>9}  {'notes'}")
    print(f"  {'-' * 24} {'-' * 9}  {'-' * 44}")
    for r in report["results"]:
        print(f"  {r['name']:<24} {r['accuracy']:>9.3f}  {r['notes']}")
        if not np.isnan(r["accuracy_high_snr"]):
            print(f"  {'  @ high SNR':<24} {r['accuracy_high_snr']:>9.3f}")
            print(f"  {f'  < {SUB_THRESHOLD_SNR_DB:g} dB':<24} {r['accuracy_sub_threshold']:>9.3f}")
        if not np.isnan(r.get("accuracy_train", float("nan"))):
            print(f"  {'  on train':<24} {r['accuracy_train']:>9.3f}")
    print()
    print(f"  {'gate check':<28} {'value':>7} {'target band':>14}   result")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 14}   {'-' * 6}")
    for c in report["checks"]:
        band = f"[{c['band'][0]:.2f}, {c['band'][1]:.2f}]"
        if not c.get("applicable", True):
            mark, shown = "n/a ", "    -- "
        else:
            mark, shown = ("PASS" if c["ok"] else "FAIL"), f"{c['value']:>7.3f}"
        print(f"  {c['check']:<28} {shown} {band:>14}   {mark}")
        if c["advice"]:
            for line in _wrap(c["advice"], 74):
                print(f"      {line}")
    print()
    print("  GATE PASSED" if report["passed"] else "  GATE FAILED -- do not train SSL methods yet")
    print()


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--label", default="emitter", choices=["emitter", "modulation"])
    p.add_argument("--n-train", type=int, default=12000)
    p.add_argument("--n-test", type=int, default=3000)
    p.add_argument(
        "--epochs",
        type=int,
        default=60,
        help=(
            "oracle training epochs. Do not lower this casually: an undertrained "
            "oracle reports 'task too hard' for a task that is merely unlearned, "
            "and the advice it prints would send you to change the wrong thing. "
            "On the easy preset, 30 epochs gives 0.81 at high SNR and 60 gives 0.89."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--no-gate",
        action="store_true",
        help="report the numbers but always exit 0 (for exploratory calibration)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_console_logging(logging.INFO)

    report = run_report(
        args.data,
        label=args.label,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
    _print_report(report)

    if args.json_out:
        write_json(args.json_out, report)
    elif (Path(args.data) / "MANIFEST.json").exists():
        write_json(Path(args.data) / f"difficulty_report_{args.label}.json", report)

    if not report["passed"] and not args.no_gate:
        print("difficulty gate failed; see advice above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
