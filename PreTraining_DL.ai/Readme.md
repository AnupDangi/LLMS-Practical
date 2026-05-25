# Pretraining LLMs (DeepLearning.AI)

**Course:** [Pretraining LLMs](https://www.deeplearning.ai/courses/pretraining-llms/)

Data preparation, packaging, training, and evaluation for causal LM pretraining.

## Prerequisites

- [Transformers_DL.ai](../Transformers_DL.ai/Readme.md)
- [Stanford CME295](../Stanford_CME295_Transformers_LLMs/Readme.md) lectures 1–4 (helpful)

## What you will learn

- Why and when pretraining matters vs. fine-tuning
- Tokenization, data prep, and training configuration
- Running training with Hugging Face `Trainer` and evaluating models

## Notebook order

| Order | Notebook | Lesson title |
| ----- | -------- | ------------ |
| 1 | [WhyPretraining.ipynb](WhyPretraining.ipynb) | Why Pretraining? |
| 2 | [Lesson_2.ipynb](Lesson_2.ipynb) | Data Preparation |
| 3 | [Lesson_3.ipynb](Lesson_3.ipynb) | Data Packaging |
| 4 | [Lesson_4.ipynb](Lesson_4.ipynb) | Preparing your model for training |
| 5 | [Lesson_5.ipynb](Lesson_5.ipynb) | Model training |
| 6 | [Lesson_6.ipynb](Lesson_6.ipynb) | Model evaluation |

**Asset:** [cost_pretraining.png](cost_pretraining.png) (referenced in Lesson 1).

## Setup

- **Packages:** `transformers`, `datasets`, `accelerate`; Lesson 6 may use `lm-eval` (see notebook `pip` cells).
- **Hardware:** GPU strongly recommended for training lessons.
- **API keys:** Hugging Face token if downloading gated models ([.env.example](../.env.example)).

## Tips

- Training lessons can take significant time; reduce batch size or steps for smoke tests.
- Notebooks reference a course `requirements.txt` that is not in this repo—use [requirements-notes.md](../requirements-notes.md).

[← Back to learning path](../README.md)
