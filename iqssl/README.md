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
| 3. Encoders, heads, EMA, `Method` contract, training loop | **done** | one loop, no method branching; `iqssl-pretrain` works |
| 4. `augment/` ops and five policies | **done** | ops, `none`/`light`/`standard`/`heavy`/`hardware_invariant`, masking, pipeline |
| 5. View methods | **done** | SimCLR, SupCon, Barlow Twins, VICReg, BYOL, SimSiam |
| 6. Masked / latent methods + references | **done** | MAE; I-JEPA/TS-JEPA/data2vec as a shared-machinery 3-axis ablation; `supervised` ceiling, `random` floor |
| 7. Full eval protocol | **done** | `iqssl-evaluate`: linear probe + kNN + finetune at 1%/10%/100%, nuisance R², SNR quartiles |
| 8. Configs + equal-budget HPO | **done** | `iqssl-sweep`: 9 trials per method, selected on a **validation** probe, budget and search space both enforced |
| 9. Analysis and aggregation | **done** | `iqssl-aggregate`: fairness-invariant pooling refusal, seed error bars, compute-vs-accuracy Pareto |

**What runs today:** the full train-and-measure path. All twelve registered
methods pretrain through the one shared loop, and `iqssl-evaluate` scores any
run — probe, kNN and finetune at three label fractions on both label axes, the
nuisance-regression probe, and accuracy by SNR quartile — writing `eval.json`
next to the checkpoint. Evaluation refuses a dataset whose hash differs from
the one the run was pretrained on.

**What does not run yet:** nothing in the pipeline. Every stage has an
implementation and a gate. What is missing is *results* — no sweep has been run
at a scale where the numbers mean anything, and the sections below say exactly
what has and has not been demonstrated.

### Tuning selects on validation, never on test

`iqssl-sweep --method simclr` runs the contracted nine trials. Two things about
it are worth knowing before trusting any tuned number:

**The objective is a validation probe, not the pretraining loss.** Hydra's
Optuna sweeper optimizes whatever `main()` returns, and returning the loss would
be worse than useless — BYOL and SimSiam reach near-zero loss precisely when
they collapse, so the sweep would reliably select the collapsed configuration
for exactly the methods designed to avoid it. `experiment=hpo` sets
`val_probe: true`, and the driver refuses to sweep an experiment that does not.

**Selection never opens the test split.** Nine trials across twelve methods
tuned against test would leak it into every headline number, invisibly.
`tests/test_hpo.py` asserts the omission by recording which splits get opened,
rather than trusting the source to keep saying "val".

The budget and the search spaces are enforced, not documented: `N_TRIALS = 9`
for every method, and a `search:` block may only reach for `base_lr`,
`weight_decay` or that method's own constructor arguments — the same hole
`merge_train_config` closes for static config, which a search space could
otherwise walk through by tuning `train.epochs`.

### Aggregation refuses to pool incomparable runs

`iqssl-aggregate --root <dir> --out <dir>` produces the tables and figures. Its
most important behaviour is a refusal: runs disagreeing on dataset hash,
encoder, epochs, batch size, warmup, grad clip, augmentation policy, crop length
or split variant are **not** averaged. That failure is silent by construction —
a table pooled across two dataset versions reads exactly like a correct one —
so `--allow-incomparable` stamps the written report, and such a table says on
its face that it is not a result.

Single-seed cells report no spread rather than zero, for the same reason: a zero
error bar is a reproducibility claim nobody measured.

### What the smoke scale can and cannot show

All twelve methods pretrain and four (SimCLR, MAE, `supervised`, `random`) were
evaluated end to end on the `smoke` preset. The protocol demonstrably resolves
signal — the **modulation** probe reads ≈0.30 against a 0.10 chance line, for
*random features* — but the **emitter** probe sits at chance (≈0.12, chance
0.125) for every method, `supervised` included.

That is a property of the preset, not a defect, and it was worth pinning down
rather than assuming. `Supervised` memorizes 64 buffers to 100% accuracy in 60
steps, so the objective and its gradient path are sound; it simply cannot
*generalize* 8-way emitter identity from 2,861 training buffers with a ViT,
which is data-starved at that size. The difficulty gate's purpose-built CNN
reached only 0.31 on the same preset.

**So do not read a ceiling-above-floor ordering off smoke runs — there isn't
one, and there shouldn't be.** That ordering is a property to check on `easy`
or larger, where the gate already measured a CNN at 0.891. Smoke exists to
exercise the plumbing in CI, and it does that well.

One cost worth knowing before a full sweep: the protocol finetunes at three
label fractions on both label axes, so six full finetunes per run — 72 across a
twelve-method sweep, which can exceed the pretraining it evaluates. Reporting
both axes is deliberate, but budget for it.

### The `easy`-scale viability gate: undertrained, not unlearnable

`experiment=easy_long` asks the question that governs whether a ~324-run sweep is
worth building toward: on data the difficulty gate certified (a SmallCNN reaches
0.891 here), does the benchmark's own encoder, loop and eval protocol reproduce
that? Measured at 5,600 steps × 128 = 717k samples, single seed:

| | emitter probe @100% | modulation probe @100% |
| --- | --- | --- |
| `supervised` | 0.103 | 0.541 |
| `random` | 0.092 | 0.523 |
| chance | 0.0625 | 0.100 |

**The ceiling does not clear the floor on either axis.** Supervised training buys
~0.01 over an untrained encoder — within seed noise. A twelve-method sweep at
this budget would produce a table in which every method scores about the same as
random features, which is precisely the outcome the gate exists to prevent.

But the training curve says *undertrained*, not *unlearnable*, and the
distinction is the whole point of logging train accuracy:

```
step     0-3600 : loss 2.773 (= ln 16), train_acc 0.062 (= chance)   -- flat
step  4000-5200 : loss 2.765 -> 2.680,  train_acc 0.072 -> 0.105     -- descending
```

The model sat at chance for 3,600 steps, began learning around step 4,000, and
the budget ended 1,600 steps later with the curve still bending. Train accuracy
(0.105) tracks test (0.103), so it is underfitting — more data would not help,
more optimization might. Extrapolating the inflection, something like 20k steps
(~5 h on four CPU cores) would be needed to find out.

**Do not read a method ranking off this preset yet, and do not launch the full
sweep until a ceiling clears its floor here.** The open question is whether the
ViT gets there with more steps or whether the control encoder should become
`cnn1d` — which the difficulty gate's CNN result mildly favours, at ~5.8x the
cost per step (measured: 0.185 steps/s against the ViT's 1.08, because the ViT's
stride-16 patch embedding discards 15/16 of the sequence before any attention
runs, while the CNN stem processes it at full resolution).

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
iqssl-evaluate --run outputs/smoke_cpu/simclr/seed0/<timestamp>

iqssl-aggregate --root outputs/smoke_cpu --out results/smoke
```

Sweep across methods and seeds (`-m` is Hydra's multirun):

```bash
# 1. tune each method on its own equal budget (nine trials, selected on val)
for m in simclr supcon barlow vicreg byol simsiam mae ijepa tsjepa data2vec supervised; do
  iqssl-sweep --method $m --experiment hpo
done

# 2. three seeds at each winner, then evaluate and aggregate
iqssl-pretrain -m experiment=main_comparison \
  method=simclr,supcon,barlow,vicreg,byol,simsiam,mae,ijepa,tsjepa,data2vec,supervised,random \
  seed=0,1,2
iqssl-aggregate --root outputs/main_comparison --out results/main
```

`random` is absent from the tuning loop deliberately: an untrained encoder has
no hyperparameters, and nine trials would measure nothing nine times.

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

**The shipped `base_lr` values are provisional.** Every `base_lr` in `configs/method/*.yaml` is a
published default lifted from a paper that tuned it at ImageNet scale — batch 4096, natural images —
and none has been validated at batch 128 on 1-D RF buffers. They are starting points for the sweep,
not tuned values, and no comparison between two methods is fair until both have been through it.
That they are guesses is not a footnote: the easy-scale viability gate spent 3,600 steps pinned at
uniform output and began learning only once cosine decay had cut the LR ~50× below its peak.
`supervised` is simply the one whose failure is legible, because it has a train accuracy to watch;
a badly-tuned contrastive run just yields a mediocre representation and says nothing.

Equal epochs is not equal compute, so every run logs `tokens_seen`, encoder forward passes,
wall-clock and peak memory, and the results include a compute-vs-accuracy Pareto plot.

Evaluation always reports frozen linear probe **and** kNN **and** finetune at 1%/10%/100%: masked
methods are known to linear-probe poorly and finetune well, so a linear-probe-only headline would
just rediscover a known artifact.

## License

No license is granted. All rights reserved — see the repository root `README.md`.
