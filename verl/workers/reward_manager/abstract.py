# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu

RawRewardFn = Callable[..., Any]


class AbstractRewardManager(ABC):
    @abstractmethod
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        pass

    @abstractmethod
    def __call__(
        self,
        data: TensorDict,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        pass

    def _extract_reward_from_rm_scores(
        self, data: TensorDict, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any] | None:
        """
        Extract reward from already-computed rm_scores if available.
        This has been deprecated.

        Args:
            data: TensorDict object containing the batch data
            return_dict: Whether to return a dictionary with reward_tensor and reward_extra_info

        Returns:
            If rm_scores exists:
                - If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
                - If return_dict=False: torch.Tensor of rm_scores
            If rm_scores doesn't exist: None
        """
        if "rm_scores" not in data.keys():
            return None

        if return_dict:
            reward_extra_keys = tu.item(data, "reward_extra_keys", [])
            reward_extra_info = {key: tu.item(data, key) for key in reward_extra_keys}
            return {"reward_tensor": data["rm_scores"], "reward_extra_info": reward_extra_info}
        else:
            return data["rm_scores"]
