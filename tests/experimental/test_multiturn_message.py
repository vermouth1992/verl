"""
Prototype multiturn message data structure
"""

from dataclasses import dataclass
from typing import Dict, List

from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from verl.tools.base_tool import OpenAIFunctionToolSchema


@dataclass
class MultiTurnMessage:
    prompt_message_list: List[Dict[str, str]]  # without system prompt
    response_message_list: List[Dict[str, str]]
    tokenizer: PreTrainedTokenizer
    tools: List[OpenAIFunctionToolSchema]
    # input_ids
    prompt_ids: List[List[int]]  # turn level
    response_ids: List[List[int]]  # turn level
    response_mask: List[int]  # turn level

    def __post_init__(self):
        """We either have valid messages list and compute ids, or we have valid ids and compute prompt messages

        Returns:

        """
        pass


def convert_list_multiturn_message_to_batch(multiturn_messages: List[MultiTurnMessage]) -> TensorDict:
    pass


if __name__ == "__main__":
    user_0 = """We define a new math operator @, where you can only call an external tool to compute. Please put your final answer inside \boxed{} only in the last turn. What is 5 @ (3 @ 6 @ 8)?"""

    assistant0_0 = """Alright, let's break this down step by step using the custom operator '@'. We need to compute the innermost expression first and then use that result for the next computation.

First, let's compute (3 @ 6 @ 8). We will use the tool to perform the calculations.

<tool_call>
{"name": "calculate", "arguments": {"a": 3, "b": 6, "operand": "@"}}
</tool_call>"""

    tool0_0 = """<tool_response>
-3
</tool_response>"""

    assistant0_1 = """We have computed that 3 @ 6 = -3. Now we need to compute (-3 @ 8).

Let's use the tool again to perform this calculation.

<tool_call>
{"name": "calculate", "arguments": {"a": -3, "b": 8, "operand": "@"}}
</tool_call>"""

    tool0_1 = """<tool_response>
-25
</tool_response>"""

    assistant0_2 = """We have computed that -3 @ 8 = -25. Now we need to compute the final expression 5 @ (-25).

Let's use the tool to perform this final calculation.

<tool_call>
{"name": "calculate", "arguments": {"a": 5, "b": -25, "operand": "@"}}
</tool_call>"""

    tool0_2 = """<tool_response>
65
</tool_response>"""

    assistant0_3 = """We have computed that 5 @ (-25) = 65.

Therefore, the final answer is \boxed{65}."""

    multiturn_messages_0 = [
        {"role": "user", "content": user_0},
        {"role": "assistant", "content": assistant0_0},
        {"role": "tool", "content": tool0_0},
        {"role": "assistant", "content": assistant0_1},
        {"role": "tool", "content": tool0_1},
        {"role": "assistant", "content": assistant0_2},
        {"role": "tool", "content": tool0_2},
        {"role": "assistant", "content": assistant0_3},
    ]

    multiturn_messages_1 = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": ""},
    ]
