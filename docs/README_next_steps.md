# verl next steps

## Introduction
We observe a wide range of adoption of verl in various RL research papers and production systems over the last 6 months. And most of them built on top of verl to develop advanced and customized features such as asynchronous RL, advanced data sampling, replay policies, multi-modal RL including audio, image and video, etc,.
We acknowledge that it is impossible for verl to adopt all these great features. So, instead of building verl as a monolithic repo, we would like to make verl a **composable** and **customizable** **library** so that people can easily build on top of it (e.g., import core components and start their own repo).
Moreover, we believe that a **multi-backend** system with well-defined APIs is essential in the long term for easy integration of different training and inference systems.
In light of these statements, we propose the following approach.

## Approach
A RL library can be decomposed into 5 parts: 1) Rollout Engine; 2) Model Engine; 3) Weight Transfer Engine; 4) Agent Loop and 5) Data Transfer System. 
Each component acts as a **service**, whose backend is agnostic to the RL system. RL system is a single controller that manipulates the data flow of those service and add customized algorithm-specific components such as replay buffer.

### Rollout Engine
Recent RL infrastructures adopt native server mode rollouts (e.g., [slime](https://github.com/THUDM/slime)). This makes integration of new features in inference engine easily and provide a great abstraction between rollout engine and the rest of the RL systems. This also makes supporting multi-backend rollout system easy as the common abstraction of various inference backend simply becomes a http endpoint. verl is migrating to this design.

### Model Engine
Similar to rollout engine, verl will also make model engine as a **service**. In order to do so, a well-defined interface is necessary. Also, it is important to make the model engine extensible to more frontier model architectures such as Qwen3-Omni, VLA and even diffusion models. By defining a model engine, high level trainers (e.g., SFT/DPO/RM) and workers (e.g., Actor/Critic) can reuse most of the code by simply changing the loss function and the data loader. Please refer to https://github.com/vermouth1992/verl/blob/chi/dev/roadmap/docs/README_model_engine.md for more details.


### Weight Transfer Engine
We will completely abandon the idea that places the rollout and model in the same process. Model engine and rollout engine have to expose APIs such that weight transfer can be performed with a backend-agnostic engine with both Cuda-IPC and NCCL. See https://github.com/MoonshotAI/checkpoint-engine for more details.

### Agent Loop
Agent framework (e.g., [SWE-Agent](https://github.com/SWE-agent/SWE-agent)) typically works in the style of OpenAI Gym fashion, which is basically a simple loop
```python
import env_lib
env = env_lib.make(env_id)
obs, done = env.reset()  # prompt
while not done:
    response = llm.call(obs)
    action = extract_action(response)
    next_obs, reward, done, info = env.step(action)  # tool call or env interaction
    obs = next_obs
```
Agent loop in verl is the **interface** that connects agent/env framework and the RL training framework. Most agent framework interacts with LLM using standard OpenAI Compatible server via string in and string out. This will cause issues in RL training as tokenizer decode + encode is not revertible. Thus, users have to handle
- Token in token out
- Convert trajectory into a format that can be consumed by the trainer

in the customized Agent Loop.

We will write a detailed instruction about how to extend to new environment/agent framework shortly.

### Data Transfer System
One of the biggest challenge of single controller programming abstraction is the data transfer overhead between services. 
To tackle this, community has proposed a [TransferQueue](https://github.com/TransferQueue/TransferQueue) prototype. The core idea is to decouple data management from data storage, enabling services to only pass data reference while actual data is fetched directly point-to-point. 
It enables sample-level data routing across the entire post-training system, preserving the flexibility of a single-controller architecture while minimizing data transfer overhead. verl will adopt this design once the prototype is ready.

