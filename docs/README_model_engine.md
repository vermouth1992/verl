# verl next steps

## Introduction
We observe a wide range of adoption of verl in various RL research papers and production systems. And most of them built on top of verl to develop advanced and customized features such as asynchronous RL, advanced data sampling and replay policies, multi-modal RL including audio, image and video, etc,.
We acknowledge that it is impossible for verl to adopt all these great features. So, instead of building verl a monolithic repo, we would like to make verl a **composable** and **customizable** **library** so that people can easily build on top of it. 
Moreover, we believe that a multi-backend system with well-defined APIs is essential in the long term for easy customization of training and inference systems.
In light of these statements, we propose the following approach.

## Approach
We decompose a RL library into 4 parts: 1) Rollout Engine; 2) Model Engine; 3) Weight Transfer Engine and 4) Agent Loop. 

### Rollout Engine
Recent RL infrastructures adopt native server mode rollouts (e.g., [slime](https://github.com/THUDM/slime)) that places a clear abstraction between the rollout and the rest components in a RL system.


### Model Engine
| Backends       | Model Supported          | Scalability                | Model Definition               | Pain points                                         |
|----------------|--------------------------|----------------------------|--------------------------------|-----------------------------------------------------|
| FSDP + ulysses | Day 1 support HF model   | - Dense is OK <br> - MoE is bad | Huggingface + monkey patch     | Monkey patch can be easily impacted by transformers version |
| MCore          | Very limited             | Best                       | GPTModel (One model for all)   | Supporting new models is extremely hard             |

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

