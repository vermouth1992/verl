# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic batch dispatch containers for TensorDict and TransferQueue metadata."""

import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import ray
from tensordict import TensorDict

from verl.utils.tensordict_utils import concat_tensordict
from verl.utils.transferqueue_utils import BatchMeta, KVBatchMeta

PADDING_SIZE_KEY = "_padding_size_key_x123d"


class _BatchPaddingConfigMeta(type):
    _enabled = False

    @property
    def auto_padding(cls) -> bool:
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._enabled

    @auto_padding.setter
    def auto_padding(cls, enabled: bool) -> None:
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        cls._enabled = enabled


class BatchPaddingConfig(metaclass=_BatchPaddingConfigMeta):
    """Global switch for padding batches to a worker-group size before dispatch."""


@dataclass
class BatchFuture:
    """Lazy collection of worker futures that materializes a batch on demand.

    ``collect_fn`` records the intended reduction. ``get`` concatenates whatever
    concrete batch type the futures resolve to, then optionally dispatches one
    partition back out.
    """

    collect_fn: Callable
    futures: list[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: list[ray.ObjectRef]) -> "BatchFuture":
        """Wrap worker futures that should be concatenated when collected."""
        return BatchFuture(collect_fn=concat_tensordict, futures=data)

    def chunk(self, chunks: int) -> list["BatchFuture"]:
        """Create futures that each select one chunk after the full batch is collected."""
        futures = []
        for i in range(chunks):

            def dispatch_fn(value, i, chunks):
                return BatchData(value).chunk(chunks=chunks)[i]

            futures.append(
                BatchFuture(
                    collect_fn=self.collect_fn,
                    dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks),
                    futures=self.futures,
                )
            )
        return futures

    def get(self):
        """Resolve the futures, concatenate them, and apply a pending dispatch."""
        output = ray.get(self.futures)
        if isinstance(output[0], TensorDict):
            output = concat_tensordict(output)
        else:
            output = type(output[0]).concat(output)
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)
        return output


class BatchData:
    """Uniform dispatch wrapper for batch data operations.

    All type-specific logic is centralized here so callers never need to branch
    on the concrete batch type.
    """

    def __init__(self, data):
        self._data = data

    def is_chunkable(self) -> bool:
        """Return whether the wrapped value can be split across workers."""
        return isinstance(self._data, self._chunkable_types())

    def is_concatable(self) -> bool:
        """Return whether the wrapped list can be reduced to one batch."""
        data = self._data
        if not isinstance(data, list | tuple) or len(data) == 0:
            return False
        return isinstance(data[0], self._concatable_types())

    def chunk(self, chunks: int):
        """Split the wrapped batch into ``chunks`` pieces along the batch dimension."""
        data = self._data
        if isinstance(data, TensorDict):
            from verl.utils.tensordict_utils import chunk_tensordict, contiguous

            raw_chunks = chunk_tensordict(data, chunks)
            return tuple(contiguous(val).consolidate() for val in raw_chunks)
        if isinstance(data, KVBatchMeta):
            from verl.utils.transferqueue_utils import kv_batch_meta2batch_meta

            data = kv_batch_meta2batch_meta(data)
        return data.chunk(chunks=chunks)

    def concat(self):
        """Concatenate the wrapped list into one batch of the original type."""
        data = self._data
        if not data:
            raise ValueError("Cannot concatenate an empty list of data items.")
        sample = data[0]
        if isinstance(sample, ray.ObjectRef):
            return BatchFuture.concat(data)
        if isinstance(sample, TensorDict):
            return concat_tensordict(data)
        if isinstance(sample, BatchMeta):
            merged_extra_info = {}
            for meta in data:
                for key, value in meta.extra_info.items():
                    merged_extra_info.setdefault(key, []).append(value)
                meta.extra_info = {}
            batch_meta = BatchMeta.concat(data)
            batch_meta.extra_info = merged_extra_info
            from verl.utils.transferqueue_utils import batch_meta2kv_batch_meta

            return batch_meta2kv_batch_meta(batch_meta)
        return type(sample).concat(data)

    @classmethod
    def _chunkable_types(cls):
        return (BatchFuture, TensorDict, KVBatchMeta, BatchMeta)

    @classmethod
    def _concatable_types(cls):
        return (ray.ObjectRef, TensorDict, KVBatchMeta, BatchMeta)
