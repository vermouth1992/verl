from dataclasses import dataclass
import torch

@dataclass
class LLMWorkerConfig:
    freeze: bool
    model_path: str
    trust_remote_code: False
    use_remove_padding: False
    use_liger: False
    enable_gradient_checkpointing: False

    micro_batch_size: int
    micro_max_token_len: int
    default_loss_function: Callable
    optimizer: xxx


@dataclass
class FSDPConfig:
    fsdp_size: int
    ulysses_sequence_parallel_size: int = 1
    model_dtype: None