# llm-from-scratch-lab

This repository collects my personal implementations, exercises, and
experiments carried out while learning how Large Language Models work
internally, by building one step by step from scratch.

The material is based on the book *"Build a Large Language Model
(From Scratch)"* by Sebastian Raschka, which I am currently reading
and studying. The code here represents my own attempts, notes, and
solutions to the exercises proposed in the book — written for learning
purposes as I progress through the chapters.

## test/ — build, fine-tune and run the trained models

This folder is **self-contained**: it lets you create the base models, download
the matching pretrained GPT-2 weights, fine-tune them for the various tasks,
save them, and then load and run them whenever you want — without opening the
`lab/` chapters.

### Files
- **`model_utils.py`** — utility module gathering everything the two programs
  need: the GPT architecture (`GPTModel`, `MultiHeadAttention`, …), pretrained
  weight download/loading, text generation (`generate_text_simple`, `generate`),
  the datasets and training loops for classification and instruction tuning, and
  the model save/load + **naming convention**.
- **`train_model.py`** — builds / fine-tunes and saves a model for a chosen task.
- **`test_model.py`** — interactive runner: pick a task and a size, it loads the
  matching trained model and runs it in an input/output loop.
- **`commands.txt`** — ready-made bash commands for every parameter combination.
- **`trained_model/`** — where all trained models are saved and looked up.

### Saved-model naming convention
Every model is saved as `<task>_<size>.pth` inside `trained_model/`, where
`<size>` is one of `small`, `medium`, `large`, `XL`:

| Task | File name |
|------|-----------|
| Simple text generation | `gen_<size>.pth` |
| Generation with temperature + top-k | `adv_gen_t<temp>_k<k>_<size>.pth` |
| Spam classifier | `spam_cls_<size>.pth` |
| Sentiment analysis | `sent_cls_<size>.pth` |
| Instruction following | `instr_<size>.pth` |

Examples: `gen_small.pth`, `adv_gen_t0.7_k3_XL.pth`, `spam_cls_medium.pth`.
`test_model.py` rebuilds the exact same name from your choices to find the file.

### Typical workflow
```bash
# 1) create a model (examples — see commands.txt for all combinations)
python train_model.py --task gen      --size small --non_interactive
python train_model.py --task spam_cls --size small --epochs 5 --trainable_blocks 1 --non_interactive
python train_model.py --task instr    --size medium --epochs 2 --non_interactive

# 2) run it
python test_model.py
```

### Notes and design choices
- **Generation vs advanced generation.** Temperature and top-k are *decoding-time*
  options, not weights, so `gen` and `adv_gen` store the same network. `adv_gen`
  simply saves it under a name that records the default preset, so you can keep
  several presets side by side. If `test_model.py` cannot find the exact
  `adv_gen_*` file, it falls back to the plain `gen_<size>.pth` of the same size.
- **Classification fine-tuning** follows the repository approach: freeze the
  whole model, replace the output head with a 2-class linear layer, and unfreeze
  the final norm plus the last `trainable_blocks` transformer blocks
  (default 1, as in the lab).
- **Instruction fine-tuning** uses the Alpaca-style prompt format and the padded,
  shifted targets with padding masked in the loss. The optional
  `--mask_instruction` flag additionally masks the instruction/input tokens so
  the loss is computed on the response only.
- **Datasets and weights** are downloaded automatically on first use
  (GPT-2 weights into `gpt2/`, SMS spam / IMDB / instruction data into the folder).
- Models run on **GPU if available** (`cuda`), otherwise CPU.

### Requirements
```bash
pip install torch tiktoken tensorflow numpy pandas tqdm matplotlib
```

## Purpose
- Reinforce my understanding of the internals of LLMs (tokenization,
  attention, transformer blocks, pretraining, fine-tuning).
- Practice implementing each component from scratch.
- Keep track of my learning progress with runnable code and notes.

## Disclaimer
This is a learning project. The concepts and structure follow the book,
while the implementations, comments, and experiments are my own work
produced during study.

## Attribution and Acknowledgements

A substantial portion of the code in this repository is derived from the
official code examples accompanying the book **"Build a Large Language Model
(From Scratch)"** by **Sebastian Raschka** (Manning Publications). The
original code is released under the **Apache License 2.0**:

```
Copyright (c) Sebastian Raschka, under Apache License 2.0 (see LICENSE.txt).
```

### Nature of the modifications

This repository does **not** contain the original materials in their original
form. The original code was provided as Jupyter notebooks; I have **adapted and
modified** it for my own personal learning purposes, specifically:

- converted the original `.ipynb` notebooks into standalone Python (`.py`) scripts;
- split notebooks into separate files per section, so each exercise can be run
  independently;
- added explanatory comments (including the notebooks' original markdown text,
  preserved as comments);
- inserted the necessary import statements at the top of each file to make
  scripts self-contained.

During the notebook-to-Python conversion, the per-file copyright headers present
in the original source were inadvertently lost. To restore proper attribution and
to comply with the terms of the Apache License 2.0, each adapted file carries the
following notice at the top:

```
# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.
```

### Purpose and scope

This material is shared **solely for my own study and as a personal learning
portfolio**. It is **not** intended as a replacement for, or a redistribution of,
the original book or its official code, and it is **not** intended for any
commercial use.

### Acknowledgements

All credit for the original design, structure, and teaching approach goes to
**Sebastian Raschka**. His book is an outstanding resource, and I am grateful for
the clarity and depth of the material.

If you want to **learn how Large Language Models work from the ground up, I
wholeheartedly recommend reading the original book** rather than relying on this
adapted code:

- **Book:** *Build a Large Language Model (From Scratch)* — Sebastian Raschka,
  Manning Publications
- **Official code repository:** https://github.com/rasbt/LLMs-from-scratch

Please support the author by purchasing the book. The full license text is
available in `LICENSE.txt`.
