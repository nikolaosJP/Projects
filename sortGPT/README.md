# customGPT

Small GPT-style training repo with a tiny character-level "sort digits" dataset.
Inspired by Andrej Karpathy's nanoGPT repo found here: `https://github.com/karpathy/nanoGPT/tree/master`.

## Introduction

customGPT is a tiny, end-to-end GPT training sandbox: it generates a synthetic
character-level dataset where inputs are digit sequences and targets are their
sorted versions (e.g., `3 1 2 -> 1 2 3`), then trains a GPT-style model to learn
that mapping. The goal is clarity over scale, so you can see the entire pipeline
from data generation to sampling in one sitting.

## Project structure

At a glance: `data/sort_char/prepare.py` builds the dataset, `train.py` runs
training with `config/train_sort_char.py`, `model.py` defines the GPT, and
`sample_char.py` loads checkpoints to generate completions.

## Setup

You need Python 3.10+, PyTorch (CUDA recommended), and NumPy:

```
uv pip install torch numpy
```

## Create the dataset

Generate the synthetic dataset (digit sequences mapped to sorted targets):

```
python data/sort_char/prepare.py
```

This writes `train.bin`, `val.bin`, `meta.pkl`, and `input.txt` under
`data/sort_char/`.

## Train the model

Use the small training config:

```
python train.py config/train_sort_char.py
```

Checkpoints and logs are written to `out-sort-char/` by default. You can
override any config value inline:

```
python train.py config/train_sort_char.py --batch_size=32 --max_iters=2000
```

## Sample / run the model

After training, sample from the checkpoint:

```
python sample_char.py --out_dir=out-sort-char --start="3 1 2 0 2 ->"
```

## Notes

`train.py` defaults to OpenWebText settings, but this repo currently only
includes the `sort_char` dataset and config. If you change
`data/sort_char/prepare.py`, re-run it to regenerate binaries.
