"""Reference estimators for the difficulty gate.

Three of them, bracketing the range in which an SSL comparison carries any
information:

``classical``
    Closed-form RF statistics into a logistic regression. If these already
    identify the emitter, every SSL method scores ~99% and the ranking is noise.
``raw_linear``
    A linear probe straight on raw IQ. Near chance is the healthy outcome; well
    above it means the label is readable without representation learning at all.
``supervised_cnn``
    A small supervised CNN, the *ceiling*. Too low and even a supervised oracle
    cannot learn the task, so method rankings would be noise for the opposite
    reason.

The bands were not chosen a priori -- they are where the gate stops being able to
distinguish "this method is better" from "this dataset is broken". The first time
this ran it failed, and diagnosing the failure found two real defects in the
generator, which is the entire argument for having it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from iqssl.dsp.convert import ri_to_complex

TARGET_BANDS: dict[str, tuple[float, float]] = {
    "classical": (0.0, 0.60),
    "raw_linear": (0.0, 0.25),
    "supervised_cnn_high_snr": (0.85, 0.95),
    "supervised_cnn_0db": (0.40, 0.60),
}

HIGH_SNR_PERCENTILE = 75.0
ZERO_DB_TOLERANCE = 3.0


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    accuracy_high_snr: float = float("nan")
    accuracy_0db: float = float("nan")
    notes: str = ""
    feature_names: list[str] = field(default_factory=list)


# --- classical features -------------------------------------------------------

CLASSICAL_FEATURE_NAMES = [
    "dc_i",
    "dc_q",
    "dc_mag",
    "circularity",
    "papr_db",
    "kurtosis",
    "c40_mag",
    "cfo_est",
    "evm_proxy",
    "spectral_flatness",
    "envelope_var",
]


def classical_features(x: Tensor) -> Tensor:
    """Closed-form RF statistics from ``(B, 2, L)`` buffers. Returns ``(B, F)``.

    Every division is guarded. Degenerate input is not hypothetical -- an
    all-zero buffer appears the moment a preset is misconfigured, and the gate
    must report a bad number rather than crash inside a reciprocal.
    """
    z = ri_to_complex(x)
    eps = 1e-12

    mean = z.mean(-1)
    power = (z.real**2 + z.imag**2).mean(-1).clamp_min(eps)
    amp = z.abs()

    # Circularity |E[z^2]| / E|z|^2.
    #
    # This replaced a spectral-asymmetry feature that was *meant* to detect IQ
    # imbalance but measured something else and scored F=0.3 against emitter id.
    # A proper (circular) complex signal has E[z^2] = 0; imbalance makes it
    # improper, and this ratio tracks |nu| directly. Critically it survives an
    # unknown carrier phase: rotating z by e^{j@} scales E[z^2] by e^{2j@} and
    # leaves the magnitude alone. After the swap, F rose to 12.5.
    circularity = (z**2).mean(-1).abs() / power

    papr = 10.0 * torch.log10((amp.pow(2).amax(-1) / power).clamp_min(eps))
    kurt = (amp.pow(4).mean(-1) / power.pow(2)).clamp(0, 100)

    # Fourth-order moment: the standard modulation-discriminating cumulant. The
    # second-order one is deliberately absent -- it is |E[z^2]|/E|z|^2, which is
    # exactly `circularity` above, and a duplicated column buys nothing.
    c40 = (z**4).mean(-1).abs() / power.pow(2)

    # Coarse CFO estimate: mean phase advance between adjacent samples, in
    # cycles per sample. Conjugate-product averaging rather than a per-sample
    # angle mean, so it does not wrap at +/-pi.
    cfo = torch.angle((z[:, 1:] * z[:, :-1].conj()).mean(-1)) / (2 * torch.pi)

    evm = (amp - amp.mean(-1, keepdim=True)).pow(2).mean(-1) / power

    spec = torch.fft.fft(z, dim=-1).abs().pow(2).clamp_min(eps)
    flat = torch.exp(torch.log(spec).mean(-1)) / spec.mean(-1)

    env_var = amp.var(-1) / power

    feats = torch.stack(
        [
            mean.real,
            mean.imag,
            mean.abs(),
            circularity,
            papr,
            kurt,
            c40,
            cfo,
            evm,
            flat,
            env_var,
        ],
        dim=-1,
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# --- the reference model ------------------------------------------------------


class SmallCNN(nn.Module):
    """The supervised oracle: small, plain, and deliberately unremarkable.

    Its job is to answer "is this task learnable at all?", so it must not be so
    strong that it flatters a broken dataset, nor so weak that its failure is
    about the model. Four strided conv blocks and a linear head.
    """

    def __init__(self, n_classes: int, in_ch: int = 2, width: int = 32) -> None:
        super().__init__()
        chans = [in_ch, width, width * 2, width * 4, width * 4]
        blocks: list[nn.Module] = []
        for a, b in itertools.pairwise(chans):
            blocks += [
                nn.Conv1d(a, b, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(b),
                nn.ReLU(inplace=True),
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(chans[-1], n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.pool(self.features(x)).flatten(1))


# --- the three baselines ------------------------------------------------------


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean()) if len(y) else float("nan")


def _logreg(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=max_iter)
    with warnings.catch_warnings():
        # The raw-IQ probe is expected to sit at chance, so it legitimately never
        # converges. That is the finding, not a problem to be tuned away.
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(scaler.transform(xtr), ytr)
    return clf.predict(scaler.transform(xte))


def run_classical_baseline(
    xtr: Tensor, ytr: np.ndarray, xte: Tensor, yte: np.ndarray
) -> BaselineResult:
    ftr = classical_features(xtr).numpy()
    fte = classical_features(xte).numpy()
    pred = _logreg(ftr, ytr, fte)
    return BaselineResult(
        name="classical",
        accuracy=_accuracy(pred, yte),
        notes=f"{len(CLASSICAL_FEATURE_NAMES)} closed-form features -> logistic regression",
        feature_names=list(CLASSICAL_FEATURE_NAMES),
    )


def run_raw_linear_baseline(
    xtr: Tensor, ytr: np.ndarray, xte: Tensor, yte: np.ndarray
) -> BaselineResult:
    """Linear probe on flattened raw IQ.

    Expected to sit at chance. It is here as the floor: if a *linear* map on raw
    samples recovers the emitter, the fingerprint is not a representation-learning
    problem at all.
    """
    a = xtr.flatten(1).numpy()
    b = xte.flatten(1).numpy()
    pred = _logreg(a, ytr, b)
    return BaselineResult(
        name="raw_linear",
        accuracy=_accuracy(pred, yte),
        notes="logistic regression on flattened raw IQ",
    )


def run_supervised_cnn_baseline(
    xtr: Tensor,
    ytr: np.ndarray,
    xte: Tensor,
    yte: np.ndarray,
    snr_te: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 3e-3,
    device: str = "cpu",
    seed: int = 0,
) -> BaselineResult:
    """Train the oracle and report overall, high-SNR and 0 dB accuracy.

    ``epochs`` defaults to 60 rather than something cheaper because 30 leaves the
    model undertrained, at which point the gate reports "task too hard" for a task
    that is merely unlearned -- and the advice it prints then sends you to change
    the wrong knob entirely.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    n_classes = int(max(ytr.max(), yte.max())) + 1

    model = SmallCNN(n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(xtr) // batch_size)
    )

    xtr_d = xtr.to(dev)
    ytr_d = torch.from_numpy(ytr).long().to(dev)
    gen = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xtr_d), generator=gen)
        for s in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xtr_d[idx]), ytr_d[idx])
            loss.backward()
            opt.step()
            sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(xte), 512):
            preds.append(model(xte[s : s + 512].to(dev)).argmax(-1).cpu())
    pred = torch.cat(preds).numpy()

    high = snr_te >= np.percentile(snr_te, HIGH_SNR_PERCENTILE)
    near0 = np.abs(snr_te) <= ZERO_DB_TOLERANCE
    return BaselineResult(
        name="supervised_cnn",
        accuracy=_accuracy(pred, yte),
        accuracy_high_snr=_accuracy(pred[high], yte[high]),
        accuracy_0db=_accuracy(pred[near0], yte[near0]) if near0.any() else float("nan"),
        notes=f"SmallCNN, {epochs} epochs, {n_classes} classes",
    )
