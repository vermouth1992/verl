# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl.batch_protocol import BatchPaddingConfig
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu

# or set env var VERL_AUTO_PADDING = "1" / "true"
BatchPaddingConfig.auto_padding = True


@ray.remote
class Actor(Worker):
    def __init__(self) -> None:
        super().__init__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_BATCH)
    def add(self, data: TensorDict):
        data["a"] += self.rank
        return data


def test_auto_padding():
    ray.init(num_cpus=100)

    chunk_size = 4
    actor_cls = RayClassWithInitArgs(cls=Actor)
    resource_pool = RayResourcePool(process_on_nodes=[chunk_size], use_gpu=False)
    actor_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=actor_cls)

    # test locally first
    for test_size in range(4, 20):
        local_data = tu.batch_from_dict({"a": torch.zeros(test_size)}, {"na": np.zeros(test_size, dtype=object)})
        # print(f"before padding, local_data = {local_data}")
        padding_size = (chunk_size - (test_size % chunk_size)) if (test_size % chunk_size > 0) else 0
        local_data = tu.pad_batch(local_data, padding_size)
        # print(f"after padding, local_data = {local_data}")
        assert len(local_data) == len(local_data) + len(local_data) % chunk_size, (
            f"expecting padded length to be {len(local_data) + len(local_data) % chunk_size}, but got {len(local_data)}"
        )
        chunked = tu.chunk_tensordict(local_data, chunk_size)
        assert len(chunked) == chunk_size, f"during test_size = {test_size}, expecting {chunk_size}, got {chunked}"
        for dp in chunked:
            assert len(dp) == test_size // chunk_size + bool(test_size % chunk_size), (
                f"test size = {test_size}, expecting dp to be length of "
                f"{test_size // chunk_size + bool(test_size % chunk_size)}, but got {len(dp)}: {dp} {chunked}"
            )

    # test with RayWorkerGroup method decorated as dispatch_mode=Dispatch.DP_COMPUTE_BATCH
    data = tu.batch_from_dict({"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)})
    output = actor_wg.add(data)

    print(output["a"])
    assert len(output) == 10, "Failed in args split and padding."

    data = tu.batch_from_dict({"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output["a"])
    assert len(output) == 10, "Failed in kwargs split and padding."

    data = tu.batch_from_dict({"a": torch.zeros(1)}, {"na": np.array([str(i) for i in range(1)], dtype=object)})
    output = actor_wg.add(data)

    print(output["a"])
    assert len(output) == 1, "Failed in args split and padding."

    data = tu.batch_from_dict({"a": torch.zeros(1)}, {"na": np.array([str(i) for i in range(1)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output["a"])
    assert len(output) == 1, "Failed in kwargs split and padding."

    data = tu.batch_from_dict({"a": torch.zeros(8)}, {"na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data)

    print(output["a"])
    assert len(output) == 8, "Failed in args split and padding."

    data = tu.batch_from_dict({"a": torch.zeros(8)}, {"na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output["a"])
    assert len(output) == 8, "Failed in kwargs split and padding."

    # test data proto specific config
    BatchPaddingConfig.auto_padding = False

    data = tu.batch_from_dict(
        {"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data)
    print(output["a"])
    assert len(output) == 10, "Failed in args split and padding."

    data = tu.batch_from_dict(
        {"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data=data)
    print(output["a"])
    assert len(output) == 10, "Failed in kwargs split and padding."

    data = tu.batch_from_single_dict(
        {"a": torch.zeros(1), "na": np.array([str(i) for i in range(1)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data)

    print(output["a"])
    assert len(output) == 1, "Failed in args split and padding."

    data = tu.batch_from_single_dict(
        {"a": torch.zeros(1), "na": np.array([str(i) for i in range(1)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data=data)

    print(output["a"])
    assert len(output) == 1, "Failed in kwargs split and padding."

    data = tu.batch_from_single_dict({"a": torch.zeros(8), "na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data)

    print(output["a"])
    assert len(output) == 8, "Failed in args split and padding."

    data = tu.batch_from_single_dict({"a": torch.zeros(8), "na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output["a"])
    assert len(output) == 8, "Failed in kwargs split and padding."

    ray.shutdown()


if __name__ == "__main__":
    test_auto_padding()
