from dataclasses import dataclass, field
import torch

from typing import Callable



@dataclass
class OptimizerConfig:
    optim_class = torch.optim.AdamW
    lr: float = 1e-3
    optim_config: dict = field()


class LRSchedulerConfig:
    pass


# Maybe we can further split the config into more fine-grained configs
@dataclass
class FSDPModelConfig:
    model_path: str  # the path to the hf model
    trust_remote_code: bool  # whether trust remote code
    use_remove_padding: bool  # whether use remove padding
    use_liger: False # whether use liger kernel or not
    freeze: bool  # whether to freeze the model
    external_lib: str  # external library to register custom hf models
    override_model_config: dict  # override the huggingface model config
    default_loss_function: Callable
    optimizer: OptimizerConfig
    fsdp_size: int # the size of zero3
    ulysses_sequence_parallel_size: int = 1  # the size of ulysses sp
    model_dtype: None # dtype of the model.
    freeze: bool # whether the model will be freeze or not
    enable_gradient_checkpointing: False # whether enable gradient checkpointing or not
    use_torch_compile: bool # whether or not to use torch.compile to accelerate ops
    # training related
    micro_batch_size: int # number of batch size in each forward
    micro_max_token_len: int # number of max tokens in each forward
