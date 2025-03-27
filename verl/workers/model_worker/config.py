from dataclasses import dataclass
import torch

from typing import Callable


# Maybe we can further split the config into more fine-grained configs
@dataclass
class FSDPModelConfig:
    model_path: str  # the path to the hf model
    trust_remote_code: bool  # whether trust remote code
    use_remove_padding: bool  # whether use remove padding
    use_liger: False
    freeze: bool  # whether to freeze the model
    external_lib: str  # external library to register custom hf models
    override_model_config: dict
    default_loss_function: Callable
    optimizer: None
    fsdp_size: int
    ulysses_sequence_parallel_size: int = 1
    model_dtype: None
    freeze: bool
    enable_gradient_checkpointing: False
    micro_batch_size: int
    micro_max_token_len: int
    use_torch_compile: bool
