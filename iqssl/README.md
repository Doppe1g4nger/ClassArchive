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
| 2. `data/` + difficulty gate | **done** | `easy` preset **PASSES**: classical 0.183, raw-IQ 0.061, oracle 0.887 @ high SNR |
| 3. Encoders, heads, EMA, `Method` contract | **partial** | ViT1D/ResNet1D/EMA/heads done + tested; **training loop not yet written** |
| 4. `augment/` ops and five policies | not started | |
| 5. View methods | **2 of 5** | SimCLR + SupCon done; Barlow/VICReg losses written and tested, methods not wired |
| 6. Masked / latent methods | not started | MAE decoder written; MAE, I-JEPA, TS-JEPA, data2vec not wired |
| 7. Full eval protocol | not started | |
| 8. Configs + equal-budget HPO | not started | |
| 9. Analysis and aggregation | not started | |

**What runs today:** dataset generation, the difficulty gate, every DSP
primitive, both encoders, the EMA teacher, all five loss functions, and the
SimCLR/SupCon method objects (unit-tested against analytic references).
`iqssl-build-dataset` and `iqssl-difficulty-report` are working CLIs.

**What does not run yet:** `iqssl-pretrain`, `iqssl-evaluate` and
`iqssl-aggregate` are declared entry points with no implementation behind them.
Nothing can actually be trained end to end until `train/loop.py` and
`augment/` exist — the loop needs the augmentation pipeline to produce the
views that `ViewSpec` promises.

### Calibration state

The difficulty gate passes on `easy`. `medium` and `hard` were recalibrated
from a measured ladder but have **not** been re-verified against the gate at
full scale — do that before trusting any result from them:

```bash
iqssl-build-dataset --out data/synth_v1 --difficulty medium --n-samples 200000
iqssl-difficulty-report --data data/synth_v1
```

If `medium` lands outside its bands, the ladder measurements say which knob to
reach for: multipath is the dominant destroyer of the fingerprint (going from 1
to 2 taps roughly halves oracle accuracy), low SNR is second, and CFO costs
very little. Widening the emitter impairment spreads is the *last* resort, not
the first — it is also the fastest way to make the task trivial.

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
# CPU-only host: smaller wheels
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu

pytest -m "not slow"                           # core suite (257 tests)

iqssl-build-dataset --out data/smoke --difficulty smoke --n-samples 4096
iqssl-difficulty-report --data data/smoke      # the gate

# Not implemented yet -- see Implementation status above.
# iqssl-pretrain experiment=smoke_cpu method=simclr
# iqssl-evaluate run=outputs/smoke_cpu/simclr/seed0/<timestamp>
# iqssl-aggregate root=outputs/smoke_cpu
```

Full sweep (once stages 3-8 land):

```bash
iqssl-pretrain -m experiment=main_comparison \
  method=simclr,supcon,barlow,vicreg,byol,simsiam,mae,ijepa,tsjepa,data2vec,supervised,random \
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
