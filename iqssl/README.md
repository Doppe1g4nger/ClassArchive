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

All twelve methods pretrain and four are evaluated end to end on the `smoke`
preset. Measured with the shipped conv-stem encoder, at fraction 1.0:

| | emitter linear | knn | finetune | modulation linear | knn | finetune |
| --- | --- | --- | --- | --- | --- | --- |
| `supervised` | 0.150 | 0.149 | 0.133 | 0.569 | 0.460 | 0.704 |
| `simclr` | 0.142 | 0.146 | 0.142 | 0.576 | 0.495 | 0.687 |
| `mae` | 0.146 | 0.147 | 0.136 | 0.558 | 0.439 | 0.703 |
| **`random`** | **0.152** | **0.152** | 0.135 | **0.578** | 0.456 | 0.692 |
| chance | 0.125 | | | 0.100 | | |

**Every method ties the untrained encoder, and `random` nominally wins both
axes.** That is the correct outcome and worth stating without euphemism:
`smoke_cpu` runs `max_steps: 20`, and twenty optimizer steps is not training. The
table measures random features four times with slightly different noise.

Two things it does establish. The protocol resolves real signal — modulation at
0.578 against a 0.100 chance line — and the whole path from generation through
augmentation, all twelve objectives, checkpointing, probing and finetuning runs
without error. That is what the tier is for.

**Emitter identity is out of reach at this scale regardless of encoder.** The
earlier version of this section reached the same conclusion, but its reasoning
did not support it: it blamed a "data-starved ViT" using an encoder that could
not learn emitter identity at *any* scale, so the preset was never the
established cause. It is now. The same conv stem that reaches 0.746 on `easy`
reaches 0.150 here, so 2,861 training buffers is the binding constraint.

**Do not read a ceiling-above-floor ordering off smoke runs — there is none, and
there should not be.** That ordering is a property of `easy` or larger, where it
now measures 0.746 against 0.159 under `light`, and 0.770 against 0.159 under `standard`.

One number moved sharply and is worth recording, since it is a property of the
encoder rather than of any method: **random-feature modulation accuracy roughly
doubled**, from ≈0.30 with the linear patch embedding to 0.578 with the conv
stem. Untrained `PatchStem` features are far stronger than untrained linear-embed
ones — the same effect that lifted the `easy` floor from 0.092 to 0.159, visible
here on the axis where random features were already competent.

One cost worth knowing before a full sweep: the protocol finetunes at three
label fractions on both label axes, so six full finetunes per run — 72 across a
twelve-method sweep, which can exceed the pretraining it evaluates. Reporting
both axes is deliberate, but budget for it.

### The `easy`-scale viability gate: PASSED

`experiment=easy_long` asks the question that governs whether a ~324-run sweep is
worth building toward: on data the difficulty gate certified (a SmallCNN reaches
0.891 here), does the benchmark's own encoder, loop and eval protocol reproduce
that? Measured at 5,600 steps × 128 = 717k samples, single seed, `--skip-finetune`:

| emitter @100% | linear | kNN | | modulation @100% | linear |
| --- | --- | --- | --- | --- | --- |
| `supervised` | **0.746** | 0.758 | | `supervised` | 0.794 |
| `random` | **0.159** | 0.205 | | `random` | 0.727 |
| chance | 0.0625 | | | chance | 0.100 |

**The ceiling clears the floor by 0.587 on the emitter axis — 4.7x the floor,
and kNN agrees independently.** The pipeline measures the headline task. Nothing
here says anything about any SSL method; it says the instrument works.

**Modulation is nearly solved by untrained features** (`random` 0.727 against the
ceiling's 0.794), so emitter identity is the only axis here with real
discriminating power. A method that wins on modulation has mostly demonstrated
that random features are good.

The floor had to be re-measured, not carried over. It rose from 0.092 to 0.159
when the encoder changed, because an untrained `PatchStem` produces better random
features than an untrained linear patch embedding. Reusing the old floor would
have reported a gap of 0.654 and overstated the result by 11%.

#### It also passes under `standard`, which is what the thesis run uses

`easy_long` runs `augment: light`; `main_comparison` inherits the root default,
`standard`. Those are different experiments, so the table above did not actually
validate the configuration the comparison will use — and `policies.py` records
`renoise` cutting emitter separability from d = 13.3 to d = 3.4, so the
difference was not hypothetical. Re-run with everything else held fixed:

| | ceiling | floor | gap | ceiling `train_acc` | ceiling `snr` R² |
| --- | --- | --- | --- | --- | --- |
| `light` | 0.746 | 0.159 | 0.587 | 0.720 | 0.490 |
| **`standard`** | **0.770** | **0.159** | **0.611** | 0.596 | 0.416 |

**The gap is wider under the harder policy**, and the floor is identical to three
decimals on every field — which is the control working: `random` is untrained and
evaluation reads clean buffers, so the training policy cannot reach it. A
difference there would have meant leakage between augmentation and eval.

Two predictions failed here, both recorded because the reasoning behind them is
tempting and wrong:

* **Scaling the probe by the `train_acc` ratio predicted ~0.6.** The ceiling
  instead *rose* to 0.770 while training accuracy *fell* to 0.596. Stronger
  augmentation trades fit on the training distribution for transfer, and train
  accuracy has now failed to predict this gate four separate times.
* **SNR was expected to fall sharply and did not** — 0.490 → 0.416. The standing
  hypothesis, untested: `standard`'s `renoise` draws its target SNR from 15–30 dB
  while `easy`'s own prior is *also* 15–30 dB, and adding noise can only lower a
  buffer's SNR toward a target, never raise it. So on this preset `renoise`
  compresses the SNR distribution rather than randomizing it, making it a weak
  SNR randomizer precisely because its range was calibrated against this dataset.
  If that is right, the policy and the preset are coupled in a way that would
  also make `renoise` near-inert on `medium` (0–20 dB) and `hard` (−5–15 dB).

So the earlier reading of "SNR is not being discarded" as a live caveat was
wrong-headed. The nuisance results track the applied augmentations exactly —
`light` varies timing and phase and discards them, varies nothing about SNR and
retains it. The eval was reporting a property of the experiment, correctly.

#### How this failed first, and what the failure taught

The same experiment with the previous encoder read `supervised` 0.103 against
`random` 0.092 — a gap of 0.011, inside seed noise. The record of how that was
diagnosed is kept because three plausible explanations were wrong before the
right one, and each wrong turn is cheap to repeat.

The first reading was that the model was merely undertrained:

```
step     0-3600 : loss 2.773 (= ln 16), train_acc 0.062 (= chance)   -- flat
step  4000-5200 : loss 2.765 -> 2.680,  train_acc 0.072 -> 0.105     -- descending
```

The model sat at chance for 3,600 steps, began learning around step 4,000, and
the budget ended 1,600 steps later with the curve still bending — so the reading
was "give it 20k steps", about five hours of CPU.

That would have bought nothing. Its implied diagnosis —
learning began only once cosine decay had cut the LR ~50x below its peak, so the
LR was too high — was tested directly and **refuted**. Three 1,200-step runs at
`base_lr` 2e-4, 5e-5 and 1e-5, a 20x span, holding everything else at
`easy_long`:

```
base_lr      train_acc, steps 0 -> 1100          loss
2.0e-4       0.061 .. 0.064 .. 0.061   (flat)    2.775 -> 2.773
5.0e-5       0.057 .. 0.063 .. 0.059   (flat)    2.776 -> 2.773
1.0e-5       0.063 .. 0.063 .. 0.060   (flat)    2.774 -> 2.773
```

Every arm sits at chance for its whole budget, at a loss of ln 16 to four
figures. The encoder diagnostics are what make this more than "too few steps":

```
              step 0 -> 1100
grad_norm     1.33 -> 1.21        gradients flow; clipped every step at 1.0
enc_dead_dims 0    -> 0           nothing is saturated
enc_std_mean  0.31 -> 0.067       representation contracting
enc_rankme    49.7 -> 36.6        ...and losing rank, fastest at the highest LR
```

A model that is merely undertrained does not shed effective rank. This one is
being driven *onto* the uniform-output solution — which is the loss-minimizing
answer when the label is not recoverable from what reaches the classifier.

So the question is no longer the schedule; it is whether the signal survives the
encoder. Two candidates were put up, and two 1,200-step arms settled it — each
changing exactly one thing against the `base_lr` 2e-4 trace above:

```
arm                          train_acc 0 -> 1100      loss           steps/s
vit1d_tiny, augment=none     0.086 -> 0.059  flat     2.76 -> 2.773    1.81
cnn1d_tiny, augment=light    0.055 -> 0.228  rising   2.78 -> 2.113    5.72
```

**It is the encoder, not the augmentation.** Removing augmentation entirely
changes nothing — the no-augmentation arm reproduces the collapse signature
exactly (`enc_std_mean` 0.31 → 0.075, `enc_rankme` 49 → 35). Swapping the
encoder and keeping the augmentation learns immediately, and was still climbing
when the budget ended.

The mechanism is the one the difficulty gate already wrote down. The fingerprint
is second-order — IQ imbalance lives in `E[z²]`, PA compression in envelope
variance, phase noise in `dphi` variance — and `SmallCNN`'s docstring records
that mean-only pooling capped it at 0.77 against 0.891 with mean+std, because
"an average cannot represent a second moment". `vit1d` applies a *linear*
projection to each 16-sample patch and then pools with a CLS token.

**And `cnn1d_tiny` is 3.2x faster, not 5.8x slower.** That figure, quoted here
and in `easy_long.yaml` as a reason to stay with the ViT, compared vit1d_tiny
against cnn1d **r18** and reported the result as a fact about "the CNN". Like
for like it is 5.72 steps/s against 1.81. It is the third measurement in this
project to be wrong by comparing across configurations — after the contended-vs-
idle timing and the 1.9x that preceded it — and the only one that pointed the
work away from the answer.

The remaining question was *which* encoder, and the answer is not "use cnn1d":
`ResNet1D.forward_masked` raises `NotImplementedError`, so adopting it as the
control variable would drop MAE, data2vec, I-JEPA and TS-JEPA — a third of the
benchmark. The cheapest fix that would keep them is a finer patch, so the linear
projection spans fewer samples. That does not work either, so the fix is not a
config change:

```
encoder                        train_acc @1,200 steps    enc_std_mean    steps/s
vit1d_tiny  linear, patch 16   0.061   flat              0.31 -> 0.067      1.81
vit1d_tiny  linear, patch  8   0.064   flat              0.31 -> 0.088      0.80
vit1d_tiny  linear, patch  4   0.059   flat              0.32 -> 0.088      0.36
vit1d_tiny  conv,   patch 16   0.177   rising            0.28 -> 0.829      1.10
cnn1d_tiny                     0.228   rising            0.19 -> 0.961      5.72
```

**`vit1d` now defaults to `stem: conv`** — `models/vit1d.PatchStem`, a small CNN
applied within each patch, mean **and** std pooled into the token. That mirrors
what the difficulty gate measured to work, and it is what the linear embedding
structurally cannot do: form a second moment. The contraction still happens for
the first ~500 steps and then reverses, which is the shape the linear stem never
reaches.

Every convolution is confined to one patch, and that is correctness rather than
tidiness. A stem run across the sequence at kernel 7 over four stride-2 layers
has a 91-sample receptive field against a 16-sample patch, so MAE's *kept* tokens
would already carry the content it is asked to reconstruct — its premise that the
encoder never sees the masked input would become false while the whole suite
still passed. `tests/test_methods.py` perturbs one patch and asserts no other
token moves.

Changing the control variable is a contract-level act, so: the previous default
is preserved as `stem: linear` and both `vit1d_tiny` and `vit1d_small` were
changed together, since a smoke run that validated a different encoder than the
comparison uses would be worth nothing. `cnn1d_tiny` remains both faster and
better on this task, and remains unusable as the shared backbone.

Two measurements taken while narrowing this down, recorded so they are not
re-derived: the `easy` buffers arrive at std 0.71 with per-buffer std spanning
0.47–0.71, so there is no input-scale pathology; and the fixed sincos positional
embedding has 1.7x the per-token norm of the patch content, which is high but
within the range ViTs normally tolerate.

With the conv stem the same 5,600-step budget takes `train_acc` from 0.086 to
**0.720**, final loss 2.773 → 0.682, and `enc_std_mean` *grows* 0.28 → 1.109
where the linear stem contracted to 0.067. That is the run whose probe scores
open this section.

One detail worth keeping, because it would mislead a shorter diagnostic: the
working run also sits at chance for its first ~400 steps, and its `enc_rankme`
drops to **13.5 by step 800 — lower than the linear stem ever reached** — before
recovering to 41.0 and climbing. For the first few hundred steps a working
encoder and a broken one look alike, and briefly the working one looks worse.

**Still do not read a method ranking off this preset.** The gate says the
instrument works, not that any objective does. Twelve methods have not been run
here, and `base_lr` for all of them remains an untuned published default.

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

#### `medium` and `hard` do not pass, and they fail differently

Both gated for the first time, at the same 48k training buffers `easy` was
certified with. Oracle accuracy on emitter id, chance 0.0625:

| preset | overall | @ high SNR | @ 0 dB | on train | band 0.85–0.95 |
| --- | --- | --- | --- | --- | --- |
| `easy` | — | **0.891** | n/a | — | pass |
| `medium` | 0.319 | **0.436** | 0.145 | 1.000 | fail |
| `hard` | 0.108 | **0.149** | 0.072 | 1.000 | fail |

**These are not the same problem.**

`medium` at 0.436 is a real task — seven times chance, and almost exactly
`easy`'s 0.891 *halved*, which is what this file's own ablation predicts for
switching multipath on (`params.py`: "+ 2-tap multipath ... roughly halves it").
`medium` is the rung where `n_taps_choices` goes from `(1,)` to `(1, 2, 3)`. The
preset is behaving as designed; what fails is that `TARGET_BANDS` is a single
global dict applied at `cli/difficulty_report.py:102` with no per-preset keying,
so every rung is graded against the band measured on `easy`. **A difficulty
ladder cannot pass a gate calibrated to one of its rungs.**

`hard` at 0.149 is a different matter: barely above the 0.0625 chance line even
in its top SNR quartile. No choice of band rescues that, because there is almost
no signal left to measure — method rankings there really would be noise, which is
what the gate exists to prevent. `hard` turns multipath from optional to
guaranteed (`n_taps_choices=(2, 3, 4)`, never 1), widens delay spread to 2.5
symbols and drops SNR to −5 dB, and the ablation says multipath dominates and low
SNR is second. It is over-specified on both.

The gate's own advice — "DATA-limited, raise `--n-train`" — is locally correct
(train accuracy is 1.000) but misleading here. Quadrupling the data took `medium`
from 0.225 to 0.436, so reaching 0.85 by that route would need a dataset far
beyond what is practical, and the cause is multipath rather than sample count.
The heuristic assumes the impairments are fixed; when a preset is over-specified
it points at the wrong lever.

#### Resolution: per-rung bands, and `hard` still fails

`PRESET_BANDS` in `data/baselines.py` now gives each rung its own oracle band,
selected from the dataset's own manifest. Re-checked against the same measured
numbers:

| preset | oracle @ high SNR | band | | @ 0 dB | band | |
| --- | --- | --- | --- | --- | --- | --- |
| `easy` | 0.891 | 0.85–0.95 | **pass** | n/a | — | — |
| `medium` | 0.436 | 0.40–0.70 | **pass** | 0.145 | 0.12–0.45 | **pass** |
| `hard` | 0.149 | 0.30–0.55 | **fail** | 0.072 | 0.10–0.35 | **fail** |

The obvious hazard here is circularity — fit each band to what its preset scored
and the gate certifies everything while meaning nothing. So the *lower* bounds
are not fitted: they come from `MIN_ORACLE_CHANCE_MULTIPLE = 5`, on the argument
that twelve methods must be rankable between an untrained floor and the oracle
ceiling, and a ceiling at 2x chance leaves the whole field inside a few points of
the floor. At 16 classes that is 0.31, which is why `hard`'s band starts at 0.30
rather than wherever `hard` happens to land. Only the upper bounds encode intent:
`easy` nearly solved, `medium` clearly harder, `hard` hardest.

**`hard` therefore still fails, and should.** At 0.149 it is 2.4x chance in its
*best* SNR quartile. `tests/test_data.py::TestDifficultyBands` pins this: if a
future edit makes 0.149 pass, the band was widened to fit the preset instead of
the preset being fixed.

The classical and raw-IQ ceilings deliberately do **not** vary by rung. They ask
whether the label is cheaply readable without representation learning, and a
harder channel is no excuse for a task closed-form features can solve.

**`easy` and `medium` are usable for SSL runs. `hard` is not** — it needs weaker
impairments, and the ablation names the lever: it is the only rung that makes
multipath mandatory (`n_taps_choices=(2, 3, 4)`, never 1) *and* drops SNR to
−5 dB, and multipath is the dominant destroyer with low SNR second.

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
# 1. tune each method on its own equal budget (nine trials, selected on val).
#    --data is required: `hpo` pins no dataset, so that tuning cannot silently
#    inherit the smoke default and optimize a probe score that is chance.
for m in simclr supcon barlow vicreg byol simsiam mae ijepa tsjepa data2vec supervised; do
  iqssl-sweep --method $m --experiment hpo --data data/easy
done

# 2. three seeds at each winner, then evaluate and aggregate
iqssl-pretrain -m experiment=main_comparison \
  method=simclr,supcon,barlow,vicreg,byol,simsiam,mae,ijepa,tsjepa,data2vec,supervised,random \
  seed=0,1,2
iqssl-aggregate --root outputs/main_comparison --out results/main
```

`random` is absent from the tuning loop deliberately: an untrained encoder has
no hyperparameters, and nine trials would measure nothing nine times.

## Running it on a GPU

The full comparison is 12 methods x 3 seeds at `main_comparison` scale, ~1.2M
optimizer steps, plus 12 x 9 tuning trials. At the ~1.1 steps/s measured on four
CPU cores that is on the order of two weeks, which is why the sweep is gated
rather than merely queued. It is an overnight job on one GPU.

Switching is a config change — `train.device`, and `main_comparison` already sets
it — because the one thing that usually has to be rebuilt is already right:
`train/loop.py` moves the batch to the device *before* `ViewPipeline` runs, so
all the augmentation DSP executes on the accelerator as batched torch ops.
Dataloader workers only read a memmap and crop.

```bash
iqssl-build-dataset --out data/easy --difficulty easy --n-samples 120000
iqssl-pretrain experiment=easy_long method=supervised train.device=cuda
```

Regenerate the dataset rather than copying it. It is ~1.2 GB, gitignored, and
deterministic from its seed, and `MANIFEST.json` carries a `dataset_hash` that
`iqssl-evaluate` checks before it will score a run — so a rebuild that produces
a matching hash is a stronger guarantee than a file transfer.

Three things to know:

* **Precision is a contract setting, not a host flag.** `train.precision` is
  fp32 / bf16 / fp16, held constant across methods like batch size, and
  `METHOD_TUNABLE` refuses to let a method config set it — a bf16 method scored
  against an fp32 one measures numerical tolerance alongside objective quality.
  bf16 is the default on `main_comparison` because it needs no loss scaler. It is
  ignored on CPU whatever is asked for, so the CI smoke tier keeps exercising the
  same numerical path a real run does.
* **Apple silicon is `mps`, not `cuda`,** and some ops may lack MPS kernels.
  `resolve_device` fails with a message naming the knob rather than a kernel
  error much later; try `experiment=smoke_cpu train.device=mps` first. There is
  deliberately no automatic CPU fallback — a run that silently drops to CPU looks
  identical in its output and takes ~100x longer to say so.
* **The bottleneck will not be arithmetic.** `vit1d_small` is 384-d, 8 layers, 65
  tokens; at batch 256 a modern GPU is launch-bound, not compute-bound. The
  instinct is to raise the batch size and you cannot — it is fixed by the
  contract, and changing it changes `lr = base_lr·B/256` for every method at
  once. The lever is running several of the 144 independent runs concurrently on
  one GPU. Measure one run alone first and record that number, *then* parallelize
  for throughput: co-tenanted runs make steps/s unmeasurable, which is how four
  numbers in this project have already gone wrong.

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

**The shipped `base_lr` values are provisional for eleven of twelve methods.** Every `base_lr` in
`configs/method/*.yaml` is a published default lifted from a paper that tuned it at ImageNet scale —
batch 4096, natural images — and none has been validated at batch 128 on 1-D RF buffers. They are
starting points for the sweep, not tuned values, and no comparison between two methods is fair
until both have been through it.

`supervised` is the one exception, and its sweep is worth reading for what it does and does not
establish. Nine trials, 400 steps each, selecting on `val_probe_acc`:

```
trial    base_lr       wd   probe     knn  rankme
    0   1.25e-03    0.059  0.1384  0.1930    39.8
    1   1.61e-03    0.022  0.1331  0.2036    36.2
    2   7.04e-04    0.040  0.1380  0.1680    48.5
    3   7.61e-03    0.265  0.0756  0.0760     6.2   <- collapse
    4   1.10e-04    0.001  0.1325  0.1364    61.7
    5   1.03e-03    0.166  0.1415  0.1837    42.7   <- winner
    6   3.17e-04    0.254  0.1380  0.1482    56.2
    7   3.55e-03    0.003  0.0918  0.1298     5.9   <- collapse
    8   4.29e-04    0.005  0.1371  0.1531    54.2
```

**What it establishes:** a viable region, roughly 1e-4 to 2e-3, whose edges the sweep reliably
rejects. Both collapses sit above 3.5e-3. And it is the learning rate, not the weight decay:
trial 3 confounded them (highest LR *and* heaviest decay), but trial 7 collapsed at nearly the
lightest decay in the space while trial 6 ran decay 0.254 perfectly healthily.

`rankme` is what separates the two failure modes, which the probe score alone cannot — too high
crushes the representation (rank ~6), too low leaves it near its random initialization (rank 61.7,
the highest in the sweep). Same poor score, opposite remedies.

**What it does not establish:** that 1.03e-3 is better than second place. The seven healthy trials
span 0.1325–0.1415, single seed, no error bars. At this budget the sweep excludes bad
configurations; it does not finely rank good ones. Note also where the winner landed —
`supervised.yaml` ships 1.0e-3, so nine trials confirmed the untuned published default to within
3% rather than improving on it. The tuning stage demonstrated its machinery here, not its value.

That the shipped values are guesses is not a footnote, though. The easy-scale viability gate spent
3,600 steps pinned at uniform output, and the LR looked like the culprit for a full day before
measurement showed the encoder was.

Equal epochs is not equal compute, so every run logs `tokens_seen`, encoder forward passes,
wall-clock and peak memory, and the results include a compute-vs-accuracy Pareto plot.

Evaluation always reports frozen linear probe **and** kNN **and** finetune at 1%/10%/100%: masked
methods are known to linear-probe poorly and finetune well, so a linear-probe-only headline would
just rediscover a known artifact.

## License

No license is granted. All rights reserved — see the repository root `README.md`.
