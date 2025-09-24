# verl next steps

## Introduction
We observe a wide range of adoption of verl in various RL research papers and production systems. And most of them built on top of verl to develop advanced and customized features such as asynchronous RL, advanced data sampling, replay policies, multi-modal RL including audio, image and video, etc,.
We acknowledge that it is impossible for verl to adopt all these great features. So, instead of building verl as a monolithic repo, we would like to make verl a **composable** and **customizable** **library** so that people can easily build on top of it. 
Moreover, we believe that a **multi-backend** system with well-defined APIs is essential in the long term for easy integration of different training and inference systems.
In light of these statements, we propose the following approach.

## Approach
A RL library can be decomposed into 5 parts: 1) Rollout Engine; 2) Model Engine; 3) Weight Transfer Engine; 4) Agent Loop and 5) Data Transfer System. 
Each component acts as a **service**, whose backend is agnostic to the RL system. RL system is a single controller that manipulates the data flow of those service and add customized algorithm-specific components such as replay buffer. One of the biggest challenge of single controller programming abstraction is the data transfer overhead between services. To tackle this, community has proposed a [TransferQueue](https://github.com/TransferQueue/TransferQueue) prototype. The core idea is to decouple data management from data storage, enabling services to only pass data reference while actual data is fetched directly point-to-point. It enables sample-level data routing across the entire post-training system, preserving the flexibility of a single-controller architecture while minimizing data transfer overhead. verl will adopt this design once the prototype is ready.

### Rollout Engine
Recent RL infrastructures adopt native server mode rollouts (e.g., [slime](https://github.com/THUDM/slime)). This makes integration of new features in inference engine easily and provide a great abstraction between rollout engine and the rest of the RL systems. This also makes supporting multi-backend rollout system easy as the common abstraction of various inference backend simply becomes a http endpoint.

### Model Engine
Similar to rollout engine, we will also make model engine as a **service**. In order to do so, a well-defined interface is necessary. Also, it is important to make the model engine extensible to more frontier model architectures such as Qwen3-Omni, VLA and even diffusion models.



We summarize the current support matrix in the table above. Although FSDP (Zero3) can in theory be used to train any models, it suffer from poor performance (e.g., MoE models) and limited context length. 


#### APIs
- Constructor: all the model engines accept 4 parameters 
- Checkpoint

![Model Engine Checkpoint System](https://github.com/vermouth1992/verl-data/blob/master/images/verl-ckpt.png)

#### Advantages
- **Accuracy can be easily verified.** In general 
- **Maximum reusability.** Note that a type of model engine only defines the input and output. By passing the loss function and input data, a single model engine can be used for various trainers
  - ModelWithLMHead can be used for Pretrain/SFT/DPO/RL(actor)
  - ModelWithValueHead can be used for RL(value)/RM

A tentative model engine API can be found: https://github.com/volcengine/verl/blob/main/verl/workers/engine/base.py#L24

### Weight Transfer Engine
We will completely abandon the idea that places the rollout and model in the same process. Model engine and rollout engine have to expose APIs such that weight transfer can be performed with a backend-agnostic engine with both Cuda-IPC and NCCL.

https://github.com/MoonshotAI/checkpoint-engine

### Agent Loop

