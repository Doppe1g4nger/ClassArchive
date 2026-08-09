# Frozen results

Measurements that cost hours of compute and cannot be recovered from the
repository alone. Everything else under `results/` is a working file: ignored,
regenerable, and expected to differ between machines.

The distinction is not tidiness. Runs happen in ephemeral containers that get
reclaimed without warning, so a number that exists only in `outputs/` exists
only until the container dies. Copying it here is what makes it a result rather
than a transient.

## What is here

| path | what it cost |
| --- | --- |
| `tuned/*.json` | 99 HPO trials, nine per method across eleven methods |
| `gates/gate_{medium,hard}.json` | the difficulty-ladder certifications |
| `easy_comparison/<method>/seed<s>/` | one pretrain + one full evaluation each |
| `easy_comparison/done.txt` | the runner's ledger |

## What is deliberately not here

**Weights.** `checkpoint.pt` is 273 MB per run against 44 KB for everything
else that run produced, and it is reconstructible from the committed
`config_full.yaml` plus the seed. `.gitignore` keeps `*.pt` out even inside this
directory, so a stray `cp -r` cannot quietly add a gigabyte.

**Runs that were not banked.** A killed pretraining leaves a directory holding a
periodic checkpoint that looks complete from the outside. Only paths recorded in
`done.txt` at bank time are copied here, so a partial run cannot enter the
frozen tree wearing a finished run's name.

## Reading a run directory

`eval.json` holds the protocol grid — {linear probe, kNN, finetune} x {1%, 10%,
100% labels} x {emitter, modulation}, plus nuisance R² and accuracy by SNR
quartile. `metrics.csv` holds the per-step loss and `rankme` traces, which is
where training pathologies are visible: `barlow/seed0` diverges there long
before its accuracy shows it. `config_full.yaml` is the resolved configuration,
including the tuned values that run actually used.
