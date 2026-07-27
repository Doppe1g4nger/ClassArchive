"""Every numeric tolerance in the test suite, in one auditable place.

Scattering magic numbers across test files makes it impossible to answer "how
accurate is the AWGN model, really?" without reading the whole suite. When a
test starts failing marginally, the fix is a considered edit *here* — with the
reason recorded — not a quiet loosening at the call site.
"""

from __future__ import annotations

# --- Pulse shaping -----------------------------------------------------------
RRC_VS_FREQ_DOMAIN = 1e-6
"""Time-domain RRC taps vs. an independent frequency-domain construction."""

RRC_NYQUIST_ISI = 1e-6
"""|RC(kT)| at nonzero symbol lags. Truncation to a finite span sets the floor."""

RRC_ENERGY = 1e-6

# --- Noise and power ---------------------------------------------------------
SNR_MEASURE_DB = 0.15
"""Measured minus requested SNR over 200 buffers of L=4096. Dominated by the
finite-sample variance of the power estimate, not by model error."""

SNR_REALIZED_DB = 0.3
"""Stored ``snr_realized`` vs. recomputation from stored parameters. Looser than
the above because it rides on top of fading and PA gain."""

NOISE_VARIANCE_SPLIT = 0.03
"""Relative error on per-component noise variance. Catches the factor-of-2 bug
where sigma is applied per-component instead of per-complex-sample."""

POWER_ESTIMATE_REL = 0.02

# --- Impairments -------------------------------------------------------------
IRR_DB = 0.2
"""Measured image-rejection ratio vs. the analytic |mu/nu|^2."""

DC_OFFSET_DBC = 0.1

SALEH_IDENTITY = 1e-3
"""Saleh output vs. input as IBO -> infinity (the nonlinearity must vanish)."""

SALEH_SCALE_INVARIANCE = 1e-4
"""Doubling input amplitude at fixed IBO must give the same output up to scale.
This is the test that catches PA strength depending on upstream normalization."""

PHASE_NOISE_VAR_REL = 0.05
"""Var(theta[n]-theta[0]) vs. 2*pi*linewidth*Ts*n over 2000 realizations."""

# --- Resampling and delay ----------------------------------------------------
FRAC_DELAY_ROUNDTRIP = 1e-3
"""Delay by +d then -d, compared over the interior (edges are unrecoverable)."""

FRAC_DELAY_PEAK_SAMPLES = 0.02
"""Parabolically-interpolated cross-correlation peak vs. the requested delay."""

RESAMPLE_TONE_BINS = 1.0

# --- Channel -----------------------------------------------------------------
RAYLEIGH_KS_P = 0.01
"""Kolmogorov-Smirnov p-value floor for |h| ~ Rayleigh."""

TAP_POWER_REL = 0.03
"""E|h_l|^2 vs. the profile power p_l, over many realizations."""

RICIAN_K_REL = 0.05

# --- Modulation --------------------------------------------------------------
CPM_CONSTANT_MODULUS = 1e-6
"""|s[n]| must be 1 for GFSK/CPFSK/MSK. Any RRC leaking into the CPM path
breaks this immediately, which is exactly what it is there to catch."""

CPM_PHASE_CONTINUITY = 1e-6

MSK_VS_CPFSK = 1e-6
"""MSK must equal CPFSK at h=0.5 with a REC pulse, sample for sample."""

CONSTELLATION_UNIT_POWER = 1e-6

# --- End-to-end --------------------------------------------------------------
CLEAN_PATH_BER = 0.0
"""Zero bit errors through the clean path with matched filtering and perfect
timing. The single best end-to-end test of the linear modulator chain."""

SNR_LEAK_CORR = 0.05
"""|corr(buffer power, SNR)| after normalization. Above this, absolute power is
a shortcut feature and every 'SNR robustness' result is contaminated."""

# --- Training ----------------------------------------------------------------
LOSS_TRACE_EXACT = 0.0
"""Two seeded runs must produce bit-identical loss traces under deterministic
algorithms. Any drift means unseeded state leaked into the step."""

RESUME_EQUIVALENCE = 1e-5
"""10 steps -> save -> resume -> 10 steps vs. an uninterrupted 20-step run.
Catches optimizer/scheduler/EMA/RNG state bugs, which otherwise show up only as
a method quietly scoring a bit worse than it should."""

EMA_MATH = 1e-6

OVERFIT_LOSS_DROP = 0.60
"""Fraction the loss must fall from its first-10-step mean during tiny-overfit."""

MIN_FEATURE_STD = 1e-3
"""Per-dim std floor. Below this the representation has collapsed."""

MIN_RANKME = 4.0
"""Effective-rank floor for a non-collapsed representation on a tiny run."""
