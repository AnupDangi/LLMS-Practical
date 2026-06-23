# LLMs Practical — Learning Path

Work bottom-up: foundations → architecture → training → applied production.

Each course folder has a **Readme.md** with notebook order, setup, and API keys. DeepLearning.AI notebooks often reference a `requirements.txt` that is not checked into this repo—see [requirements-notes.md](requirements-notes.md) for inferred installs per course.

**If you want to learn fundamentals refer these repos**

## KrishNaik: ML,DL,Data Science and NLP: [link](https://github.com/AnupDangi/DS_ML_DL_NLP_BOOTCAMP)

## AndrewNG: Machine Learning Specialization: [link](https://github.com/AnupDangi/Machine-Learning-Specialization/)

---

## Getting started

1. **Python 3.12+** (see [pyproject.toml](pyproject.toml)).
2. From the repo root:
   ```bash
   uv sync
   # or: python -m venv .venv && source .venv/bin/activate && pip install -e .
   ```
3. Copy [.env.example](.env.example) to `.env` and add keys for courses that call APIs (OpenAI, Hugging Face, etc.).
4. Open the course **Readme.md** before running notebooks in that folder.

---

## 1. Foundations — Backprop & Autograd

**Folder:** [`micrograd/`](micrograd/) — [Readme](micrograd/Readme.md)

Build a tiny autograd engine from scratch. Understand how gradients flow before touching any framework.

| Order | File                                                                     |
| ----- | ------------------------------------------------------------------------ |
| 1     | [Backprop Part A](micrograd/micrograd_lecture_first_half_roughly.ipynb)  |
| 2     | [Backprop Part B](micrograd/micrograd_lecture_second_half_roughly.ipynb) |
| 3     | [Full Demo](micrograd/demo.ipynb)                                        |
| (opt) | [From scratch](micrograd/micrograd_from_scratch.ipynb)                   |

---

## 2. Character-Level Language Models

**Folder:** [`makemore/`](makemore/) — [Readme](makemore/Readme.md)

Build language models from scratch: bigrams → MLP → deeper nets.

| Order | File                                             |
| ----- | ------------------------------------------------ |
| 1     | [Bigrams](makemore/makemore_part1_bigrams.ipynb) |
| 2     | [MLP](makemore/MLP.ipynb)                        |
| 3     | [Deeper MLP](makemore/makemorepart3.ipynb)       |

---

## 3. Transformers

**Folder:** [`Transformers_DL.ai/`](Transformers_DL.ai/) — [Readme](Transformers_DL.ai/Readme.md)

DeepLearning.AI course on Transformer internals — tokenizers, attention, architecture.

| Order | File                                              |
| ----- | ------------------------------------------------- |
| 1     | [L2 — Tokenizers](Transformers_DL.ai/L2.ipynb)    |
| 2     | [L6 — Full Pipeline](Transformers_DL.ai/L6.ipynb) |

---

## 4. Stanford CME295 — Transformers & LLMs

**Folder:** [`Stanford_CME295_Transformers_LLMs/`](Stanford_CME295_Transformers_LLMs/) — [Readme](Stanford_CME295_Transformers_LLMs/Readme.md)

Graduate lecture PDFs (no notebooks). Read in order; use the cheatsheet as a quick reference.

- [Cheatsheet PDF](Stanford_CME295_Transformers_LLMs/cheatsheet-transformers-large-language-models.pdf)
- Lectures 1–9: `fall25-cme295-lecture1.pdf` … `fall25-cme295-lecture9.pdf`

---

## 5. Pretraining

**Folder:** [`PreTraining_DL.ai/`](PreTraining_DL.ai/) — [Readme](PreTraining_DL.ai/Readme.md)

DeepLearning.AI course on pretraining LLMs — data, compute, training loop.

| Order | File                                                      |
| ----- | --------------------------------------------------------- |
| 1     | [Why Pretraining](PreTraining_DL.ai/WhyPretraining.ipynb) |
| 2     | [Data Preparation](PreTraining_DL.ai/Lesson_2.ipynb)      |
| 3     | [Data Packaging](PreTraining_DL.ai/Lesson_3.ipynb)        |
| 4     | [Prepare for Training](PreTraining_DL.ai/Lesson_4.ipynb)  |
| 5     | [Model Training](PreTraining_DL.ai/Lesson_5.ipynb)        |
| 6     | [Model Evaluation](PreTraining_DL.ai/Lesson_6.ipynb)      |

---

## 6. Fine-tuning

**Folder:** [`Finetuning_DL.ai/`](Finetuning_DL.ai/) — [Readme](Finetuning_DL.ai/Readme.md)

End-to-end fine-tuning: instruction tuning, data prep, training, evaluation.

| Order | File                                                                            |
| ----- | ------------------------------------------------------------------------------- |
| 1     | [Why Fine-tuning](Finetuning_DL.ai/01_Why_finetuning_lab_student.ipynb)         |
| 2     | [Where it fits](Finetuning_DL.ai/02_Where_finetuning_fits_in_lab_student.ipynb) |
| 3     | [Instruction tuning](Finetuning_DL.ai/03_Instruction_tuning_lab_student.ipynb)  |
| 4     | [Data prep](Finetuning_DL.ai/04_Data_preparation_lab_student.ipynb)             |
| 5     | [Training](Finetuning_DL.ai/05_Training_lab_student.ipynb)                      |
| 6     | [Evaluation](Finetuning_DL.ai/06_Evaluation_lab_student.ipynb)                  |

---

## 7. Post-training: Fine-tuning & RL

**Folder:** [`Fine-tuning & RL for LLMs: Intro to Post-training_DL.ai`](Fine-tuning%20%26%20RL%20for%20LLMs%3A%20Intro%20to%20Post-training_DL.ai/Readme.md)- [Readme]

RLHF, GRPO, reward modeling, evaluation, and production. Five modules.

Work through Module 1 → 5.

---

## 8. Build & Train LLM with JAX

**Folder:** [`BuildandTrainLLMwithJax_DL.ai/`](BuildandTrainLLMwithJax_DL.ai/) — [Readme](BuildandTrainLLMwithJax_DL.ai/Readme.md)

Implement LLM training using JAX. Assumes prior knowledge from courses above.

| Order | File                                                           |
| ----- | -------------------------------------------------------------- |
| 1     | [L2 — Architecture](BuildandTrainLLMwithJax_DL.ai/L2.ipynb)    |
| 2     | [L3 — Data Loading](BuildandTrainLLMwithJax_DL.ai/L3.ipynb)    |
| 3     | [L4 — Training](BuildandTrainLLMwithJax_DL.ai/L4.ipynb)        |
| 4     | [L5 — Load Pretrained](BuildandTrainLLMwithJax_DL.ai/L5.ipynb) |

---

## 9. Building & Evaluating Advanced RAG

**Folder:** [`BuildingandEvaluatingAdvancedRAG_DL.ai/`](BuildingandEvaluatingAdvancedRAG_DL.ai/) — [Readme](BuildingandEvaluatingAdvancedRAG_DL.ai/Readme.md)

Advanced RAG pipelines, retrieval strategies, and evaluation (LlamaIndex, TruLens).

| Order | File                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------- |
| 1     | [L1 — Advanced Pipeline](BuildingandEvaluatingAdvancedRAG_DL.ai/L1-Advanced_RAG_Pipeline.ipynb)   |
| 2     | [L2 — RAG Triad Metrics](BuildingandEvaluatingAdvancedRAG_DL.ai/L2-RAG_Triad_of_metrics.ipynb)    |
| 3     | [L3 — Sentence Window](BuildingandEvaluatingAdvancedRAG_DL.ai/L3-Sentence_window_retrieval.ipynb) |
| 4     | [L4 — Auto-merging](BuildingandEvaluatingAdvancedRAG_DL.ai/L4-Auto-merging_Retrieval.ipynb)       |

---

## 10. Long-Term Agentic Memory (LangGraph)

**Folder:** [`Long-Term Agentic Memory With LangGraph_DL.ai/`](Long-Term%20Agentic%20Memory%20With%20LangGraph_DL.ai/) — [Readme](Long-Term%20Agentic%20Memory%20With%20LangGraph_DL.ai/Readme.md)

Email assistant with semantic, episodic, and procedural memory using LangGraph.

| Order | File                                                                                            |
| ----- | ----------------------------------------------------------------------------------------------- |
| 1     | [Lesson 2 — Baseline](Long-Term%20Agentic%20Memory%20With%20LangGraph_DL.ai/lesson2.ipynb)      |
| 2     | [Lesson 3 — Semantic](Long-Term%20Agentic%20Memory%20With%20LangGraph_DL.ai/lesson_3.ipynb)     |
| 3     | [Lesson 4 — + Episodic](Long-Term%20Agentic%20Memory%20With%20LangGraph_DL.ai/lesson_4.ipynb)   |
| 4     | [Lesson 5 — + Procedural](Long-Term%20Agentic%20Memory%20With%20LangGraph_DL.ai/lesson_5.ipynb) |

Lesson 1 is not included in this repo; start at Lesson 2 or watch the course videos first.

---

## 11. Building AI Voice Agents for Production

**Folder:** [`Building AI Voice Agents for Production_DL.ai/`](Building%20AI%20Voice%20Agents%20for%20Production_DL.ai/) — [Readme](Building%20AI%20Voice%20Agents%20for%20Production_DL.ai/Readme.md)

Voice agent components and latency optimization for production.

| Order | File                                                                                           |
| ----- | ---------------------------------------------------------------------------------------------- |
| 1     | [Lesson 4 — Components](Building%20AI%20Voice%20Agents%20for%20Production_DL.ai/Lesson4.ipynb) |
| 2     | [Lesson 5 — Latency](Building%20AI%20Voice%20Agents%20for%20Production_DL.ai/Lesson5.ipynb)    |

Lessons 1–3 are not in this repo; use the DeepLearning.AI course for earlier lessons.

---

## 12. Introduction to On-Device AI

**Folder:** [`Introduction to on-device AI_DL.ai/`](Introduction%20to%20on-device%20AI_DL.ai/) — [Readme](Introduction%20to%20on-device%20AI_DL.ai/Readme.md)

Deploy and quantize vision models on device (Qualcomm AI Hub). Capstone for edge deployment—not text LLMs.

| Order | File                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------ |
| 1     | [L2 — Deploy Segmentation](Introduction%20to%20on-device%20AI_DL.ai/L2_Student.ipynb)            |
| 2     | [L3 — Prepare Deployment](Introduction%20to%20on-device%20AI_DL.ai/L3_Student.ipynb)             |
| 3     | [L4 — Quantization](Introduction%20to%20on-device%20AI_DL.ai/L4_Student.ipynb)                   |
| 4     | [Appendix — Build App](Introduction%20to%20on-device%20AI_DL.ai/Appendix-Building_the_App.ipynb) |

---

## 13. Agent Memory: Building Memory-Aware Agents

**Folder:** [`Agent Memory: Building Memory-Aware Agents_DL.ai/`](Agent%20Memory:%20Building%20Memory-Aware%20Agents_DL.ai/) — [Readme](Agent%20Memory:%20Building%20Memory-Aware%20Agents_DL.ai/Readme.md)

Build agents with semantic, episodic, and procedural memory systems for long-term context retention and intelligent decision-making.

| Order | File | Lesson title |
| ----- | ---- | ------------ |
| 1 | [L2.ipynb](Agent%20Memory:%20Building%20Memory-Aware%20Agents_DL.ai/L2.ipynb) | Memory Systems Fundamentals |
| 2 | [L3.ipynb](Agent%20Memory:%20Building%20Memory-Aware%20Agents_DL.ai/L3.ipynb) | Semantic Memory & Retrieval |
| 3 | [L4.ipynb](Agent%20Memory:%20Building%20Memory-Aware%20Agents_DL.ai/L4.ipynb) | Episodic & Procedural Memory |
| 4 | [L5.ipynb](Agent%20Memory:%20Building%20Memory-Aware%20Agents_DL.ai/L5.ipynb) | Scaling & Production Patterns |

---

## 14. A2A Protocol: Agentic AI on Vertex AI

**Folder:** [`A2AProtocol_DL.ai/`](A2AProtocol_DL.ai/) — [Readme](A2AProtocol_DL.ai/Readme.md)

Build production-grade agentic AI systems using Google Vertex AI with Claude. Learn prompt caching, multi-turn conversations, and tool-based agent patterns.

| Order | File | Lesson title |
| ----- | ---- | ------------ |
| 1 | [L3.ipynb](A2AProtocol_DL.ai/L3.ipynb) | Building a QA Agent with Claude on Vertex AI |
| 2 | [L4.ipynb](A2AProtocol_DL.ai/L4.ipynb) | Prompt Caching & Cost Optimization |
| 3 | [L5.ipynb](A2AProtocol_DL.ai/L5.ipynb) | Multi-Turn Conversations |
| 4 | [L6.ipynb](A2AProtocol_DL.ai/L6.ipynb) | Advanced Tool Calling |
| 5 | [L7.ipynb](A2AProtocol_DL.ai/L7.ipynb) | Agent Workflow Patterns |
| 6 | [L8.ipynb](A2AProtocol_DL.ai/L8.ipynb) | Error Handling & Robustness |
| 7 | [L9.ipynb](A2AProtocol_DL.ai/L9.ipynb) | Performance Optimization |
| 8 | [L10.ipynb](A2AProtocol_DL.ai/L10.ipynb) | Production Deployment |

---

## 15. Agentic AI from DeepLearning.AI

## -[link](https://github.com/AnupDangi/Agentic-AI-)

## Recommended order

```
micrograd → makemore → Transformers_DL.ai → Stanford_CME295
    → PreTraining_DL.ai → Finetuning_DL.ai
    → Post-training (RL) → BuildandTrainLLMwithJax
    → Advanced RAG → LangGraph Memory → Voice Agents → Agent Memory
    → A2A Protocol → On-device AI
```

## Extra resources to follow and be updated with AI applications and research.

- https://aiengineeringfromscratch.com/
- https://www.aibuilderclub.com/
