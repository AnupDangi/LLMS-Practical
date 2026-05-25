# Per-course Python packages

DeepLearning.AI notebooks often mention a `requirements.txt` that is not in this repo. Install base deps from [pyproject.toml](pyproject.toml), then add course-specific packages below.

| Course folder | Typical extra installs |
|---------------|------------------------|
| micrograd | (base: torch, numpy, matplotlib) |
| makemore | torch |
| Transformers_DL.ai | `transformers>=4.46` |
| Stanford_CME295 | (PDFs only) |
| PreTraining_DL.ai | `transformers`, `datasets`, `accelerate` |
| Finetuning_DL.ai | `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes` (GPU) |
| Post-training RL | `transformers`, `trl`, `datasets`, `torch`, grading deps in notebook |
| BuildandTrainLLMwithJax | `jax`, `flax`, `grain`, `orbax-checkpoint` |
| Advanced RAG | `llama-index`, `llama-index-llms-openai`, `trulens`, `chromadb` or similar |
| LangGraph memory | `langgraph`, `langchain`, `langchain-openai`, `python-dotenv`, `pydantic` |
| Voice agents | see notebook imports (often `livekit`, `openai`, etc.) |
| On-device AI | `qai-hub`, `qai-hub-models`, `torch` |

Run cells that contain `!pip install ...` in each notebook if imports fail.
