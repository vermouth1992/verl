Data interface
=========================

Last updated: 11/16/2026.

Training and rollout code exchanges one :class:`tensordict.TensorDict` batch. The schema has three kinds of fields:

- tensor fields, stored directly and indexed on the batch dimension
- per-sample Python values, stored as ``NonTensorStack``
- batch-level metadata, stored as ``NonTensorData``

Per-sample values move with their row during index, chunk, and repeat operations. Batch-level metadata stays with the batch. Concatenation keeps the first batch's metadata. Metrics that must include every worker are collected explicitly before concatenation.

TensorDict
~~~~~~~~~~~~

A TensorDict is a dict-like container for tensors. To instantiate one, specify the key-value pairs and the batch size.

.. code-block:: python

    >>> import torch
    >>> from tensordict import TensorDict
    >>> batch = TensorDict({"zeros": torch.zeros(2, 3, 4), "ones": torch.ones(2, 3, 5)}, batch_size=[2])
    >>> batch["twos"] = 2 * torch.ones(2, 5, 6)

Helpers in :mod:`verl.utils.tensordict_utils` construct mixed batches, select rows, concatenate nested tensors, and preserve metadata. For the underlying container, see the official tensordict_ documentation.

.. _tensordict: https://pytorch.org/tensordict/stable/overview.html
