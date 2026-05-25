[![Course overview](image.png)](https://www.deeplearning.ai/courses/introduction-to-on-device-ai/)

# Introduction to On-Device AI (DeepLearning.AI)

**Course:** [Introduction to on-device AI](https://www.deeplearning.ai/courses/introduction-to-on-device-ai/)

Deploy and quantize vision models for mobile/edge devices using Qualcomm AI Hub—not text LLM training.

## Prerequisites

- General ML familiarity ([PreTraining_DL.ai](../PreTraining_DL.ai/Readme.md)-level comfort with PyTorch helps)
- Prior sections in this repo are LLM-focused; this course is a **deployment capstone** on edge AI

## What you will learn

- On-device segmentation deployment workflows
- Model preparation and export for devices
- Quantization for efficient inference
- Optional app integration (appendix)

## Notebook order

| Order | Notebook | Lesson title |
| ----- | -------- | ------------ |
| 1 | [L2_Student.ipynb](L2_Student.ipynb) | Deploying Segmentation Models On-Device |
| 2 | [L3_Student.ipynb](L3_Student.ipynb) | Preparing for on-device deployment |
| 3 | [L4_Student.ipynb](L4_Student.ipynb) | Quantizing Models |
| 4 | [Appendix-Building_the_App.ipynb](Appendix-Building_the_App.ipynb) | Building the App |

**Not in this repo:** Lesson 1. Use the course site for the opening lesson.

## Setup

- **Packages:** `qai-hub`, `qai-hub-models`, `torch` (see [requirements-notes.md](../requirements-notes.md)).
- **Hardware:** Some steps target Qualcomm AI Hub / device toolchains; L2 and L4 notebooks are large downloads.
- **API keys:** Qualcomm AI Hub account if notebooks call hosted compile/deploy APIs (follow course signup).

## Tips

- Kernel startup ~30s (noted in course).
- [FFNet paper](https://arxiv.org/abs/2206.08236) linked from L2 for model background.
- This path ends the repo’s linear curriculum; it applies deployment thinking to models you may use alongside LLMs in products.

[← Back to learning path](../README.md)
