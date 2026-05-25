[![Course overview](image.png)](https://www.deeplearning.ai/courses/how-transformer-llms-work/)

# How Transformer LLMs Work (DeepLearning.AI)

**Course:** [How Transformer LLMs Work](https://www.deeplearning.ai/courses/how-transformer-llms-work/)

Hands-on Transformer internals: tokenizers and a full model pipeline.

## Prerequisites

- [makemore](../makemore/Readme.md) or similar PyTorch + language modeling basics
- Comfort with tensors and training loops

## What you will learn

- How trained LLM tokenizers behave (BPE, special tokens)
- Transformer building blocks and inference-style pipelines
- Using Hugging Face `transformers` Auto classes

## Notebook order

| Order | Notebook | Lesson title |
| ----- | -------- | ------------ |
| 1 | [L2.ipynb](L2.ipynb) | Comparing Trained LLM Tokenizers |
| 2 | [L6.ipynb](L6.ipynb) | Model Example (full pipeline) |

Only L2 and L6 are stored in this repo; other lessons are on the course platform.

## Setup

- **Packages:** `transformers>=4.46`, `torch` (see [requirements-notes.md](../requirements-notes.md)).
- **Hardware:** GPU optional; some downloads are large.
- **API keys:** Usually not required for tokenizer / local model demos; follow each notebook.

## Tips

- L2 downloads tokenizer assets; ensure network access on first run.
- Ignore `!pip install` cells if your environment already satisfies imports.

[← Back to learning path](../README.md)
