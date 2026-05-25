[![Course overview](image.png)](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax/)

# Build and Train an LLM with JAX (DeepLearning.AI)

**Course:** [Build and Train an LLM with JAX](https://www.deeplearning.ai/courses/build-and-train-an-llm-with-jax/)

Implement architecture, data loading, training, and inference with JAX / Flax.

## Prerequisites

- [PreTraining_DL.ai](../PreTraining_DL.ai/Readme.md) and [Finetuning_DL.ai](../Finetuning_DL.ai/Readme.md)
- Comfort with Transformer concepts from [Transformers_DL.ai](../Transformers_DL.ai/Readme.md)

## What you will learn

- LLM architecture in JAX/Flax NNX
- Data pipelines with Grain
- Training loop and loading pretrained checkpoints

## Notebook order

| Order | Notebook | Lesson title |
| ----- | -------- | ------------ |
| 1 | [L2.ipynb](L2.ipynb) | Building the LLM Architecture |
| 2 | [L3.ipynb](L3.ipynb) | Data Loading with Grain |
| 3 | [L4.ipynb](L4.ipynb) | Training the Model |
| 4 | [L5.ipynb](L5.ipynb) | Loading and Running a Pre-trained LLM |

## Setup

- **Packages:** `jax`, `jaxlib`, `flax`, `grain`, `orbax-checkpoint` (see [requirements-notes.md](../requirements-notes.md) and notebook `pip` cells).
- **Hardware:** GPU/TPU recommended for training; L5 may download large checkpoints.
- **API keys:** Usually not required.

## Tips

- Kernel startup can take ~30s on first open (noted in course UI).
- JAX installs differ by platform (CUDA vs. CPU)—match your machine when installing.

[← Back to learning path](../README.md)
