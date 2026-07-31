"""Run collection, fairness-invariant enforcement, tables and figures.

The last stage, and the last line of defence. Everything upstream works to keep
runs comparable — one training loop, one evaluation protocol, an enforced tuning
budget — and all of it can still be undone here by pooling two runs that were
never comparable in the first place. A results table computed over a mixture of
dataset versions, encoder widths or augmentation policies looks exactly like a
correct one.

So :mod:`iqssl.analysis.invariants` refuses to pool before
:mod:`iqssl.analysis.tables` gets a chance to average.
"""

from iqssl.analysis.collect import collect_runs
from iqssl.analysis.invariants import IncomparableRuns, check_invariants
from iqssl.analysis.tables import headline_table, nuisance_table, policy_interaction_table

__all__ = [
    "IncomparableRuns",
    "check_invariants",
    "collect_runs",
    "headline_table",
    "nuisance_table",
    "policy_interaction_table",
]
