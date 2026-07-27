"""One module per SSL objective, all behind a single :class:`~iqssl.methods.base.Method` contract.

The contract is what makes the comparison fair: the training loop reads a
method's :class:`~iqssl.types.ViewSpec`, hands it a :class:`~iqssl.types.Batch`,
and takes back a :class:`~iqssl.types.MethodOutput`. It never branches on which
method it is training, so no method can quietly acquire a scheduling, batching or
evaluation advantage that another lacks.
"""

from iqssl.methods.base import Method

__all__ = ["Method"]
