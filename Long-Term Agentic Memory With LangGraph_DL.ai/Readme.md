[![Course overview](image.png)](https://www.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph/)

# Long-Term Agentic Memory With LangGraph (DeepLearning.AI)

**Course:** [Long-Term Agentic Memory With LangGraph](https://www.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph/)

Build an email assistant with increasing memory capabilities: baseline rules → semantic → episodic → procedural memory.

## Prerequisites

- [Finetuning_DL.ai](../Finetuning_DL.ai/Readme.md) or comfort with LLM APIs
- [BuildingandEvaluatingAdvancedRAG_DL.ai](../BuildingandEvaluatingAdvancedRAG_DL.ai/Readme.md) (helpful for retrieval/memory concepts)

## What you will learn

- LangGraph agents and state graphs
- Classifying and responding to emails with tools
- Semantic, episodic, and procedural memory stores

## Notebook order

| Order | Notebook | Lesson title |
| ----- | -------- | ------------ |
| 1 | [lesson2.ipynb](lesson2.ipynb) | Baseline Email Assistant |
| 2 | [lesson_3.ipynb](lesson_3.ipynb) | Email Assistant with Semantic Memory |
| 3 | [lesson_4.ipynb](lesson_4.ipynb) | Semantic + Episodic Memory |
| 4 | [lesson_5.ipynb](lesson_5.ipynb) | Semantic + Episodic + Procedural Memory |

**Not in this repo:** Lesson 1 (course intro / setup). Watch on the course site or download the full notebook bundle before Lesson 2.

## Setup

- **Packages:** `langgraph`, `langchain`, `langchain-openai`, `python-dotenv`, `pydantic` (see [requirements-notes.md](../requirements-notes.md)).
- **Hardware:** CPU OK; API latency dominates.
- **API keys:** `OPENAI_API_KEY` in `.env` ([.env.example](../.env.example)).

## Tips

- **Missing assets:** Notebooks reference `prompts.py` and `img/memory_course_email.png` from the full course download—they are not in this repo snapshot. Copy them from the DeepLearning.AI lab files if imports or images fail.
- AI outputs vary between runs (noted in the course).
- Run each lesson after the previous; memory patterns build on the same email-assistant design.

[← Back to learning path](../README.md)
