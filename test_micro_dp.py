"""
Test case to show how to perform loss reduction in gradient accumulation and data parallel. We test 3 cases
1. Single GPU
2. DDP/Zero 1 and the number of micro batch is different
3. Zero3 with FSDP

torchrun

"""

import os
from datetime import timedelta

import torch


def forward_backward_batch(model, input_ids, attention_mask):
    pass


if __name__ == "__main__":
    # initialize torch distributed

    torch.distributed.init_process_group(
        "gloo",
        timeout=timedelta(seconds=3600),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.manual_seed(42 + rank)

    # prepare model and dataset
    # model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M')
    #
    # vocab_size = model.config.vocab_size
    #
    # per_device_batch_size = 1
    # seqlen = 1024
    #
    # batch_size = per_device_batch_size * world_size
    # input_ids = torch.randint(0, vocab_size, (batch_size, seqlen))
    #
    # # create random attention mask
    # attention_mask = create_random_mask(input_ids, max_ratio_of_left_padding=0.3,
    #                                     max_ratio_of_valid_token=0.8, min_ratio_of_valid_token=0.5)
    #
    # # make sure the data is identical in each rank
    #
    # # slice the data of the current rank
    #
    # # create a list of micro batches
    #
    # # perform model update on each micro batch
