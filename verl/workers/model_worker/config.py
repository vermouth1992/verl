from dataclasses import dataclass
import torch

from typing import Callable



@dataclass
class ModelConfig:
    model_path: str  # the path to the hf model
    trust_remote_code: bool  # whether trust remote code
    use_remove_padding: bool  # whether use remove padding
    freeze: bool  # whether to freeze the model
    external_lib: str  # external library to register custom hf models
    override_model_config: dict
    default_loss_function: Callable

    optimizer: None


@dataclass
class FSDPConfig:
    fsdp_size: int
    ulysses_sequence_parallel_size: int = 1
    model_dtype: None
    freeze: bool
    
    use_liger: False
    enable_gradient_checkpointing: False

    micro_batch_size: int
    micro_max_token_len: int
