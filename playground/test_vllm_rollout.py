import asyncio

import ray
from omegaconf import OmegaConf

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager

if __name__ == "__main__":
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/home/chi/models/SmolLM2-135M-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
    config.trainer.n_gpus_per_node = 1

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    # =========================== 1. Init rollout manager ===========================
    async_rollout_manager = init_async_rollout_manager(config)

    # test sleep and wake_up
    async_rollout_manager.sleep()
    async_rollout_manager.wake_up()

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",
            }
        ],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    # batch = DataProto(
    #     non_tensor_batch={
    #         "raw_prompt": np.array(raw_prompts),
    #     },
    # )
    # result = async_rollout_manager.generate_sequences(prompts=batch)
    #
    # # check result
    # seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    # assert len(result) == 2
    # assert result.batch["input_ids"].size(1) == seq_len
    # assert result.batch["attention_mask"].size(1) == seq_len
    # assert result.batch["position_ids"].size(1) == seq_len

    sampling_params = dict(top_p=1, temperature=1)

    async def run_generate(raw_prompt):
        result_2 = async_rollout_manager.submit_chat_completions(messages=raw_prompt, sampling_params=sampling_params)
        await result_2

    async def generate():
        tasks = []
        for raw_prompt in raw_prompts:
            tasks.append(asyncio.create_task(run_generate(raw_prompt)))

        await asyncio.gather(*tasks)

    asyncio.run(generate())
