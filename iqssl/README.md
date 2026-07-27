# IQSSL

A controlled benchmark for **self-supervised representation learning on IQ signal buffers**
(complex baseband RF).

The deliverable is not ten method implementations — it is a harness in which the dataset, encoder,
schedule, seeds, and evaluation protocol are held fixed, per-method tuning gets an explicitly equal
budget, and the only free variable is the SSL objective.

## Methods

| family | methods |
| --- | --- |
| contrastive | SimCLR, **SupCon** (supervised control) |
| redundancy reduction | Barlow Twins |
| variance–invariance–covariance | VICReg |
| negative-free distillation | BYOL, SimSiam |
| masked input reconstruction | MAE |
| masked **latent** prediction | I-JEPA, TS-JEPA, data2vec |
| references | `supervised` (ceiling), `random` (floor) |

The three JEPA variants are built as a deliberate 3-axis ablation — target depth × context
construction × predictor — so the comparison isolates *why* latent prediction works, not just
whether it does.

## The synthetic dataset

Real RF corpora bake their perturbations in opaquely, so you cannot ask what a representation
discarded. This generator synthesizes signals from a parametric perturbation model and **retains the
ground-truth value of every nuisance parameter it applied**.

```
bits → symbol mapping → pulse shaping ─┐
                                       ├→ emitter impairments → channel → AWGN → crop → normalize
       CPM phase accumulation ─────────┘   (fixed per device)   (per sample)
```

- **Linear modulations** — BPSK, QPSK, 8PSK, 16QAM, 64QAM, PAM4, OOK (RRC-shaped)
- **Continuous-phase modulations** — GFSK, CPFSK, MSK (phase accumulation, *never* RRC-shaped)
- **Emitter fingerprint** (fixed per device) — IQ gain/phase imbalance, DC offset, Saleh PA
  nonlinearity at a sampled input back-off, oscillator phase-noise linewidth
- **Channel nuisance** (resampled per buffer) — multipath TDL (Rayleigh/Rician), CFO, phase,
  sample-rate offset, fractional timing delay, AWGN at a target SNR

Three label axes come out: modulation class, emitter id, and a continuous nuisance vector.
`data.primary_label` picks which one SupCon supervises on and the headline probe targets.

**Emitter parameters are statistically independent of channel parameters by construction.** If
carrier frequency offset partly encoded emitter identity, "did the representation discard CFO?"
would become uninterpretable — discarding it would *help* one task and hurt the other.

## Two things that decide whether the results mean anything

**1. The difficulty gate.** A per-emitter DC offset is recoverable by `mean(x)`; a gain imbalance by
`var(I)/var(Q)`. If the fingerprint task is trivially solvable, every method scores ~99% and the
comparison has no signal. So `iqssl-difficulty-report` must land in these bands before any SSL
method is trained:

| reference estimator | target |
| --- | --- |
| classical features (DC, IRR, CFO, EVM, cumulants → logistic regression) | < 60% |
| linear probe on raw IQ | < 25% |
| supervised CNN oracle @ high SNR | 85–95% |
| supervised CNN oracle @ 0 dB | 40–60% |

**2. The augmentation policy.** Contrastive methods become invariant to whatever you augment with —
and IQ imbalance, DC offset, and PA nonlinearity *are* the emitter label. The default `standard`
policy is channel-invariance only. `hardware_invariant` ships as a **positive control** that should
visibly destroy emitter accuracy; if it doesn't, the pipeline is wrong. Policy is a first-class
sweep axis, so the thesis reports the method × policy interaction rather than comparing a method
against another method plus a hand-designed prior.

## Implementation status

The framework is built in stages, riskiest and most foundational first, so that
each stage has an independently verifiable gate rather than everything landing
untested at the end.

| Stage | State | Gate |
| --- | --- | --- |
| 0. Scaffold, registry, contracts, CI | **done** | lint + types + tests green |
| 1. `dsp/` signal primitives | **done** | 113 property tests; BER=0 clean path, every digital modulation |
| 2. `data/` + difficulty gate | **done** | `easy` **PASSES** at 48k train buffers: classical 0.195, raw-IQ 0.062, oracle 0.891 @ high SNR |
| 3. Encoders, heads, EMA, `Method` contract | **partial** | ViT1D/ResNet1D/EMA/heads done + tested; **training loop not yet written** |
| 4. `augment/` ops and five policies | **done** | ops, `none`/`light`/`standard`/`heavy`/`hardware_invariant`, masking, pipeline |
| 5. View methods | **done** | SimCLR, SupCon, Barlow Twins, VICReg, BYOL, SimSiam |
| 6. Masked / latent methods | not started | MAE decoder and mask generators written; MAE, I-JEPA, TS-JEPA, data2vec not wired |
| 7. Full eval protocol | not started | |
| 8. Configs + equal-budget HPO | **partial** | Hydra tree and per-method configs exist; no HPO sweeper yet |
| 9. Analysis and aggregation | not started | |

**What runs today:** everything through pretraining. `iqssl-pretrain` trains any
of the six view methods end to end, on CPU, writing `metrics.csv`, a config
snapshot, run provenance and a checkpoint. `iqssl-build-dataset` and
`iqssl-difficulty-report` work as before.

**What does not run yet:** `iqssl-evaluate` and `iqssl-aggregate` do not exist,
so nothing yet measures a *representation* — pretraining logs an online probe as
a diagnostic, and that is not the evaluation protocol. The four masked and
latent-prediction methods are not wired, though the pieces they need (patch
decoder, `random`/`block`/`causal` mask generation, the ViT's three forward
paths) are built and tested.

### A note on the gitignore incident

`iqssl/data/` was absent from the first four commits on this branch. `.gitignore`
carried an unanchored `data/`, which git matches against a directory of that name
at *any* depth, so the whole package — generator, splits, storage, baselines —
was silently excluded. Nothing complained: `git add` reported no new files,
`git status` was clean, and the local suite passed because the modules were on
disk the entire time. It surfaced only in CI, after the session that wrote them
had ended and the container had been reclaimed, and the code had to be rewritten
from its surviving tests.

That is why the four directory rules are now anchored (`/data/`, `/outputs/`,
`/multirun/`, `/results/`) and why `tests/test_scaffold.py` asks git directly
whether any `iqssl/**/*.py` is ignored. Do not un-anchor them.

### Calibration state

**The gate PASSES on `easy`**, and the rewritten generator reproduces the
calibration recorded before the package was lost:

```bash
iqssl-build-dataset --out data/easy --difficulty easy --n-samples 120000
iqssl-difficulty-report --data data/easy --n-train 48000 --n-test 6000
```

| check | measured | recorded | band |
| --- | --- | --- | --- |
| classical | 0.195 | 0.183 | < 0.60 — **pass** |
| raw-IQ linear | 0.062 | 0.061 | < 0.25 — **pass** |
| oracle @ high SNR | **0.891** | **0.887** | 0.85–0.95 — **pass** |

All three land within about 0.01 of the earlier numbers, and raw-IQ sits at
chance for 16 emitters, which is the healthy result. That agreement is the main
evidence that the rewrite is faithful rather than merely self-consistent.

**Sample count is the binding constraint, and diagnosing that took a
detour worth recording.** At the gate's default 12k training buffers the oracle
reached only 0.785, and *both* obvious fixes point the wrong way: train accuracy
was 1.000, so the oracle was memorizing, which means strengthening the
impairments would have made the task easier and lengthening the buffer would
have treated a signal weakness that was not the constraint. 48k training buffers
took it to 0.891 with no other change. The gate now reports train accuracy and
branches its advice on it, so the diagnosis is automatic.

Two changes were made before that diagnostic existed:

- The emitter prior was widened once (0.632 → 0.771 at 12k). It was reasoning
  from the wrong model of the failure, but the final numbers vindicate the
  setting — classical 0.195 against a recorded 0.183 and the oracle within 0.004
  of its recorded value. Strictly, this leaves one thing unmeasured: whether the
  *narrow* prior would also pass at 48k. The 12k trend (0.632 narrow vs 0.771
  wide) suggests not, but that is an inference, not a measurement.
- `SmallCNN` pools mean **and** standard deviation rather than mean alone
  (0.771 → 0.785 at 12k). Kept because a ceiling should be able to express the
  second-order statistics its own classical floor uses, not because it closed
  the gap — it did not.

`medium` and `hard` have **not** been verified against the gate. Do that before
trusting any result from them, and start from 48k training buffers, since both
are strictly harder than `easy`:

```bash
iqssl-build-dataset --out data/synth_v1 --difficulty medium --n-samples 200000
iqssl-difficulty-report --data data/synth_v1 --n-train 48000
```

If a preset lands outside its bands, read the train accuracy first. When the
oracle is genuinely underfitting rather than memorizing, the ladder says which
knob to reach for: multipath is the dominant destroyer of the fingerprint (1 to
2 taps roughly halves oracle accuracy), low SNR is second, and CFO costs very
little. Widening the emitter spreads is the *last* resort — it is the fastest
way to make the task trivial.

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
# CPU-only host: smaller wheels
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu

pytest -m "not slow"                           # core suite

iqssl-build-dataset --out data/smoke --difficulty smoke --n-samples 4096
iqssl-difficulty-report --data data/smoke --no-gate   # the gate; see calibration below

iqssl-pretrain experiment=smoke_cpu method=simclr     # ~20 steps on CPU

# Not implemented yet -- see Implementation status above.
# iqssl-evaluate run=outputs/smoke_cpu/simclr/seed0/<timestamp>
# iqssl-aggregate root=outputs/smoke_cpu
```

Sweep across methods and seeds (`-m` is Hydra's multirun):

```bash
iqssl-pretrain -m experiment=main_comparison \
  method=simclr,supcon,barlow,vicreg,byol,simsiam \
  seed=0,1,2
```

## Layout

```
dsp/       pure batched complex64 primitives — shared by generation and augmentation
data/      composes dsp into the generative distribution; records every nuisance
augment/   wraps the same dsp primitives as train-time view transforms
models/    encoders (held fixed across methods) and heads
methods/   one module per objective, all behind a single Method contract
train/     the shared loop — it never branches on method type
eval/      one protocol, applied identically to every method
analysis/  run collection, fairness-invariant enforcement, tables and figures
```

`dsp/` is separate from **both** `data/` and `augment/` on purpose: shared primitives, two
pipelines, no coupling between the data distribution and the view distribution.

## The fairness contract

**Held constant:** encoder architecture and init seed; dataset, splits, and the shared 1%/10% label
subsets; batch size; epochs; 10% warmup and cosine decay; grad clip; the `lr = base_lr·B/256`
scaling rule; seeds {0,1,2}; the entire eval protocol; and the HPO budget — 9 trials per method,
then 3 seeds at the winner.

**Deliberately varies:** optimizer family (LARS / SGD / AdamW, per published defaults — forcing one
would handicap several methods), tuned LR and weight decay, projector and predictor dimensions, and
method-specific coefficients. **Fix the encoder, not the heads**: Barlow Twins and VICReg genuinely
need wide projectors, so capping them at 128-d would be a handicap dressed up as fairness.

Equal epochs is not equal compute, so every run logs `tokens_seen`, encoder forward passes,
wall-clock and peak memory, and the results include a compute-vs-accuracy Pareto plot.

Evaluation always reports frozen linear probe **and** kNN **and** finetune at 1%/10%/100%: masked
methods are known to linear-probe poorly and finetune well, so a linear-probe-only headline would
just rediscover a known artifact.

## License

No license is granted. All rights reserved — see the repository root `README.md`.
