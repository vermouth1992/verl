# Model Engine

## Current Support Matrix
| Backends       | Model Supported        | Scalability                | Model Definition               | Pain points                                                 |
|----------------|------------------------|----------------------------|--------------------------------|-------------------------------------------------------------|
| FSDP + ulysses | Day 1 support HF model | - Dense is OK <br> - MoE is bad | Huggingface + monkey patch     | Monkey patch can be easily impacted by transformers version |
| MCore          | Limited                | Best                       | GPTModel (One model for all)   | Supporting new models is difficult                          |

- We monkey patch attention function to support ulysses
- We monkey patch VLM models to support FSDP with mixed data with and without images

## Class Hierarchy
Note that all the workers and trainers run in **SPMD** mode. SFT/DPO/RM trainer is directly invoked by `torchrun`. The Actor/Critic worker can also be invoked by a RayWorkerGroup and provides APIs to a single controller. 

- Base Engine level: implement model init, optimizer init, lr scheduler init, sharding, checkpoint manager.
- Full Engine level: subclass base engine and implement `forward_step`.
- Worker/SPMD trainer level: **engine agnostic**, implement training logics using the engine

RL trainer utilizes workers to construct HybridFlow program. This is out of the scope of model engine.

## Existing Model Types
| Model type | Language model         | Value model            |
|------------|------------------------|------------------------|
| Input      | text/image/video/audio | text/image/video/audio |
| Output     | logits for next token  | logits as value        |

Currently, we have two model types: language model and value model. We expect to expand the category to include Qwen-Omni family (output both text and audio) and VLA models.

## Data Format
Currently, verl adopts left-right padding data format in RL trainer. This creates massive padding when the discrepancy between response length is large. We will start to implement no-padding format throughout the whole system. 

![Data Format](https://github.com/vermouth1992/verl-data/blob/master/images/data_format.png)

Here is the migration plan:
- Implement no-padding format in engine
- Add a transformation layer in Actor/Critic worker.
- Replace Actor/Critic Worker in RL trainer
- Implement no-padding throughput system


## Checkpoint System
![Model Engine Checkpoint System](https://github.com/vermouth1992/verl-data/blob/master/images/verl-ckpt.png)

The engine constructs the model using huggingface config, then load weights from huggingface checkpoint. If the engine directly uses huggingface model definition, it can use function provided by `transformers`. Otherwise, each engine has to write their own checkpoint load logic (e.g., [mbridge](https://github.com/ISEEKYAN/mbridge)). During model training, each engine has to implement save_checkpoint and load_checkpoint that save/load intermediate sharded checkpoint including model, optimizer and lr scheduler states. Each engine has to implement a checkpoint merge script, that merges the intermediate sharded checkpoint back to huggingface format.


## API
A tentative model engine API can be found: https://github.com/volcengine/verl/blob/main/verl/workers/engine/base.py#L24

## Extension

### Add a new backend
- Start a new folder under `verl/workers/engine`. Then, implement `transformer_impl.py`. If you want to implement a non-transformer model, please contact us in advance.

### Add a new model type


