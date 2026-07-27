"""Transmitter hardware impairments — the emitter fingerprint.

These are the effects held *fixed per device*, so together they constitute the
identity a fingerprinting model has to recover. Two design rules keep them
honest as a machine-learning target:

**Parameterize by physically meaningful quantities**, not by whatever scale the
signal happens to arrive at. The Saleh PA is the sharp case: without an explicit
input back-off, distortion strength silently depends on upstream normalization,
so regenerating the dataset after an unrelated change to the pulse-shaping gain
would quietly change the task difficulty.

**Prefer forms that can be measured back out.** IQ imbalance is written in
image-rejection form because a test can transmit a tone, measure the image, and
compare against the analytic ``|mu/nu|^2``. The ad-hoc "scale I, rotate Q"
formulation computes the same thing but has no such closed-form check.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def iq_imbalance(x: Tensor, gain_db: Tensor | float, phase_deg: Tensor | float) -> Tensor:
    """Gain/phase imbalance between the I and Q branches.

    Image-rejection form ``y = mu*x + nu*conj(x)`` with
    ``mu = (1 + g e^{-j phi}) / 2`` and ``nu = (1 - g e^{j phi}) / 2``.
    Ideal hardware (``gain_db=0, phase_deg=0``) gives ``mu=1, nu=0`` exactly.
    """
    mu, nu = imbalance_coeffs(gain_db, phase_deg, device=x.device)
    while mu.ndim < x.ndim:
        mu = mu.unsqueeze(-1)
        nu = nu.unsqueeze(-1)
    return mu * x + nu * x.conj()


def imbalance_coeffs(
    gain_db: Tensor | float,
    phase_deg: Tensor | float,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    g_db = torch.as_tensor(gain_db, dtype=torch.float32, device=device)
    p_deg = torch.as_tensor(phase_deg, dtype=torch.float32, device=device)
    g = torch.pow(10.0, g_db / 20.0)
    phi = p_deg * math.pi / 180.0
    mu = (1 + g * torch.polar(torch.ones_like(g), -phi)) / 2
    nu = (1 - g * torch.polar(torch.ones_like(g), phi)) / 2
    return mu.to(torch.complex64), nu.to(torch.complex64)


def image_rejection_ratio_db(gain_db: Tensor | float, phase_deg: Tensor | float) -> Tensor:
    """Analytic IRR ``= 20*log10|mu/nu|``. The reference the tone test compares to."""
    mu, nu = imbalance_coeffs(gain_db, phase_deg)
    return 20.0 * torch.log10(mu.abs() / nu.abs().clamp_min(1e-20))


def dc_offset(x: Tensor, dc_i: Tensor | float, dc_q: Tensor | float, *, dbc: bool = True) -> Tensor:
    """Add an LO-leakage DC term.

    ``dbc=True`` interprets the arguments as dB relative to the buffer's RMS,
    which keeps the offset's *relative* size fixed regardless of signal scale —
    the same reasoning as the PA back-off.
    """
    i = torch.as_tensor(dc_i, dtype=torch.float32, device=x.device)
    q = torch.as_tensor(dc_q, dtype=torch.float32, device=x.device)
    if dbc:
        from iqssl.dsp.power import rms

        scale = rms(x, keepdim=True)
        i = torch.pow(10.0, i / 20.0)
        q = torch.pow(10.0, q / 20.0)
        offset = torch.complex(i, q)
        while offset.ndim < x.ndim:
            offset = offset.unsqueeze(-1)
        return x + offset * scale
    offset = torch.complex(i, q)
    while offset.ndim < x.ndim:
        offset = offset.unsqueeze(-1)
    return x + offset


def saleh(
    x: Tensor,
    *,
    alpha_a: Tensor | float = 2.0,
    beta_a: Tensor | float = 1.0,
    alpha_p: Tensor | float = 4.0,
    beta_p: Tensor | float = 9.0,
    ibo_db: Tensor | float = 6.0,
) -> Tensor:
    """Saleh memoryless power-amplifier model with explicit input back-off.

    ``A(r) = alpha_a r / (1 + beta_a r^2)`` (AM/AM) and
    ``Phi(r) = alpha_p r^2 / (1 + beta_p r^2)`` (AM/PM, radians).

    With the classic ``alpha_a=2, beta_a=1`` the amplifier saturates at ``r=1``,
    so the signal is first scaled to sit ``ibo_db`` below that point, distorted,
    then rescaled back. Because the scaling is undone afterwards, ``ibo_db`` is
    the *only* control on distortion strength — doubling the input amplitude
    doubles the output and changes nothing else, which
    ``test_pa_ibo_scale_invariance`` pins down.
    """
    from iqssl.dsp.power import rms

    def col(v: Tensor | float) -> Tensor:
        t = torch.as_tensor(v, dtype=torch.float32, device=x.device)
        while t.ndim < x.ndim:
            t = t.unsqueeze(-1)
        return t

    aa, ba, ap, bp = col(alpha_a), col(beta_a), col(alpha_p), col(beta_p)
    backoff = torch.pow(10.0, -col(ibo_db) / 20.0)

    scale = backoff / rms(x, keepdim=True).clamp_min(1e-12)
    xs = x * scale

    r = xs.abs()
    r2 = r * r
    amp = aa * r / (1 + ba * r2)
    phase_shift = ap * r2 / (1 + bp * r2)
    ys = torch.polar(amp, torch.angle(xs) + phase_shift)

    return (ys / scale).to(torch.complex64)


def phase_noise(
    x: Tensor,
    linewidth_hz: Tensor | float,
    sample_rate_hz: float,
    *,
    random_initial_phase: bool = True,
    generator: torch.Generator | None = None,
    unit_steps: Tensor | None = None,
    initial_phase: Tensor | None = None,
) -> Tensor:
    """Oscillator phase noise as a Wiener process.

    ``theta[n+1] = theta[n] + N(0, 2*pi*linewidth/fs)``, so
    ``Var(theta[n] - theta[0]) = 2*pi*linewidth*n/fs`` — the property the test
    checks, and equivalent to a Lorentzian line of the given 3 dB width.

    The initial phase is randomized **per buffer, not per emitter**. A fixed
    per-emitter starting phase would be a trivial identity leak: the model would
    read the fingerprint straight off sample zero instead of learning the
    spectral character of the oscillator, which is the actual signature.

    ``unit_steps`` and ``initial_phase`` let a caller supply the randomness
    instead of drawing it. The dataset generator uses this so that a sample's
    signal depends only on its own RNG stream — otherwise regenerating with a
    different chunk size would silently produce a different dataset.
    """
    b, ell = x.shape
    lw = torch.as_tensor(linewidth_hz, dtype=torch.float32, device=x.device)
    if lw.ndim == 0:
        lw = lw.expand(b)
    step_std = torch.sqrt(2 * math.pi * lw / sample_rate_hz).unsqueeze(-1)

    if unit_steps is None:
        unit_steps = torch.randn(b, ell, generator=generator, device=x.device, dtype=torch.float32)
    theta = torch.cumsum(unit_steps[:, :ell] * step_std, dim=-1)
    theta = theta - theta[:, :1]  # start the walk at zero, then offset explicitly

    if initial_phase is not None:
        theta = theta + initial_phase.to(x.device).view(b, 1)
    elif random_initial_phase:
        phi0 = (
            torch.rand(b, 1, generator=generator, device=x.device, dtype=torch.float32)
            * 2
            * math.pi
        )
        theta = theta + phi0
    return x * torch.polar(torch.ones_like(theta), theta)


def apply_emitter_chain(
    x: Tensor,
    *,
    iq_gain_db: Tensor | float,
    iq_phase_deg: Tensor | float,
    dc_i_dbc: Tensor | float,
    dc_q_dbc: Tensor | float,
    pa_ibo_db: Tensor | float,
    pa_alpha_a: Tensor | float = 2.0,
    pa_beta_a: Tensor | float = 1.0,
    pa_alpha_p: Tensor | float = 4.0,
    pa_beta_p: Tensor | float = 9.0,
    pn_linewidth_hz: Tensor | float = 0.0,
    sample_rate_hz: float = 1e6,
    generator: torch.Generator | None = None,
    pn_unit_steps: Tensor | None = None,
    pn_initial_phase: Tensor | None = None,
) -> Tensor:
    """The transmitter chain in its canonical order.

    ``baseband imbalance + DC -> LO phase noise -> PA``. The order is fixed and
    documented rather than left to the caller: imbalance and DC are baseband
    effects that precede upconversion, and the PA sits last in the real signal
    path, so its compression must see the already-impaired envelope.

    ``pn_unit_steps``/``pn_initial_phase`` forward externally-drawn randomness to
    :func:`phase_noise`. The dataset generator supplies them so a sample's signal
    depends only on its own RNG stream: drawing from a shared ``generator`` here
    would make the result depend on how many samples happened to be processed
    together, and the dataset hash would stop being reproducible.
    """
    y = iq_imbalance(x, iq_gain_db, iq_phase_deg)
    y = dc_offset(y, dc_i_dbc, dc_q_dbc, dbc=True)
    y = phase_noise(
        y,
        pn_linewidth_hz,
        sample_rate_hz,
        generator=generator,
        unit_steps=pn_unit_steps,
        initial_phase=pn_initial_phase,
    )
    return saleh(
        y,
        alpha_a=pa_alpha_a,
        beta_a=pa_beta_a,
        alpha_p=pa_alpha_p,
        beta_p=pa_beta_p,
        ibo_db=pa_ibo_db,
    )
