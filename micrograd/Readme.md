[![Course overview](image.png)](https://youtu.be/VMj-3S1tku0?si=pSHylWzr9XLT2DSJ)

# micrograd — Backprop & Autograd

**Course:** [Neural Networks: Zero to Hero](https://youtu.be/VMj-3S1tku0?si=pSHylWzr9XLT2DSJ) · [micrograd (GitHub)](https://github.com/karpathy/micrograd)

Karpathy-style autograd from scratch (no PyTorch for the core engine).

## Prerequisites

None. Start here if you are new to the repo.

## What you will learn

- Scalar-valued computation graphs and local derivatives
- Backpropagation by hand and in code
- Building and training a small MLP on synthetic data

## Notebook order

| Order | Notebook | Focus |
| ----- | -------- | ----- |
| 1 | [micrograd_lecture_first_half_roughly.ipynb](micrograd_lecture_first_half_roughly.ipynb) | Backprop Part A |
| 2 | [micrograd_lecture_second_half_roughly.ipynb](micrograd_lecture_second_half_roughly.ipynb) | Backprop Part B |
| 3 | [demo.ipynb](demo.ipynb) | End-to-end demo |
| (optional) | [micrograd_from_scratch.ipynb](micrograd_from_scratch.ipynb) | Compact reimplementation |

## Setup

- **Packages:** [pyproject.toml](../pyproject.toml) (`torch`, `numpy`, `matplotlib`) is enough for most cells.
- **Hardware:** CPU is fine.
- **API keys:** Not required.

## Tips

- Run notebooks top to bottom; later cells depend on earlier definitions.
- `demo.ipynb` assumes you understand the `Value` class from the lecture notebooks.

[← Back to learning path](../README.md)
