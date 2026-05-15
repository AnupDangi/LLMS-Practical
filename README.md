# LLMs Practical — Learning Path

Work bottom-up: foundations → architecture → training → fine-tuning.

---

## 1. Foundations — Backprop & Autograd

**Folder:** [`micrograd/`](micrograd/)

Build a tiny autograd engine from scratch. Understand how gradients flow before touching any framework.

| Order | File                                                                     |
| ----- | ------------------------------------------------------------------------ |
| 1     | [Backprop Part A](micrograd/micrograd_lecture_first_half_roughly.ipynb)  |
| 2     | [Backprop Part B](micrograd/micrograd_lecture_second_half_roughly.ipynb) |
| 3     | [Full Demo](micrograd/demo.ipynb)                                        |

---

## 2. Character-Level Language Models

**Folder:** [`makemore/`](makemore/)

Build language models from scratch: bigrams → MLP → deeper nets.

| Order | File                                             |
| ----- | ------------------------------------------------ |
| 1     | [Bigrams](makemore/makemore_part1_bigrams.ipynb) |
| 2     | [MLP](makemore/MLP.ipynb)                        |
| 3     | [Deeper MLP](makemore/makemorepart3.ipynb)       |

---

## 3. Transformers

**Folder:** [`Transformers_DL.ai/`](Transformers_DL.ai/)

DeepLearning.AI course on Transformer internals — attention, positional encoding, architecture.

| Order | File                                              |
| ----- | ------------------------------------------------- |
| 1     | [L2 — Attention](Transformers_DL.ai/L2.ipynb)     |
| 2     | [L6 — Full Pipeline](Transformers_DL.ai/L6.ipynb) |

---

## 4. Stanford CME295 — Transformers & LLMs

**Folder:** [`Stanford_CME295_Transformers_LLMs/`](Stanford_CME295_Transformers_LLMs/)

8-lecture graduate course. Read lectures in order. Cheatsheet is a good quick reference.

- [Course Notes](Stanford_CME295_Transformers_LLMs/Readme.md)
- [Cheatsheet PDF](Stanford_CME295_Transformers_LLMs/cheatsheet-transformers-large-language-models.pdf)
- Lectures 1–8 in folder

---

## 5. Pretraining

**Folder:** [`PreTraining_DL.ai/`](PreTraining_DL.ai/)

DeepLearning.AI course on pretraining LLMs — data, compute, training loop.

| Order | File                                                      |
| ----- | --------------------------------------------------------- |
| 1     | [Why Pretraining](PreTraining_DL.ai/WhyPretraining.ipynb) |
| 2–6   | Lessons 2–6 in order                                      |

---

## 6. Fine-tuning

**Folder:** [`Finetuning_DL.ai/`](Finetuning_DL.ai/)

End-to-end fine-tuning: instruction tuning, data prep, training, evaluation.

| Order | File                       |
| ----- | -------------------------- |
| 1     | Why Fine-tuning (`01_`)    |
| 2     | Where it fits (`02_`)      |
| 3     | Instruction tuning (`03_`) |
| 4     | Data prep (`04_`)          |
| 5     | Training (`05_`)           |
| 6     | Evaluation (`06_`)         |

---

## 7. Post-training: Fine-tuning & RL

**Folder:** [`Fine-tuning & RL for LLMs: Intro to Post-training_DL.ai/`](<Fine-tuning & RL for LLMs: Intro to Post-training_DL.ai/>)

Covers RLHF, DPO, reward modeling. 5 modules.

Work through Module 1 → 5. Start with [Readme](Fine-tuning%20%26%20RL%20for%20LLMs:%20Intro%20to%20Post-training_DL.ai/Readme.md).

---

## 8. Build & Train LLM with JAX

**Folder:** [`BuildandTrainLLMwithJax_DL.ai/`](BuildandTrainLLMwithJax_DL.ai/)

Implement LLM training using JAX. Assumes prior knowledge from courses above.

| Order | File              |
| ----- | ----------------- |
| 1–4   | L2 → L3 → L4 → L5 |

---

## Recommended Order

```
micrograd → makemore → Transformers_DL.ai → Stanford_CME295
    → PreTraining_DL.ai → Finetuning_DL.ai
    → Post-training (RL) → BuildandTrainLLMwithJax
```
