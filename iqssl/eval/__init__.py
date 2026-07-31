"""One evaluation protocol, applied identically to every method.

The protocol is part of the fairness contract, so it is deliberately *not*
configurable: every probe hyperparameter is a module constant in
:mod:`iqssl.eval.probes`, and the CLI exposes only which run, which dataset,
which device. A config tree here would be an invitation to per-method drift —
the precise failure this benchmark exists to prevent.

Three measurements, all mandatory: frozen linear probe, kNN, and finetune, each
at 1% / 10% / 100% of labels. Masked methods are known to linear-probe poorly
and finetune well, so any single-mode headline would just rediscover a published
artifact and attribute it to the objective.
"""

from iqssl.eval.protocol import evaluate_run

__all__ = ["evaluate_run"]
