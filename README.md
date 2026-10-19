# llm-from-scratch-lab

This repository collects my personal implementations, exercises, and
experiments carried out while learning how Large Language Models work
internally, by building one step by step from scratch.

The material is based on the book *"Build a Large Language Model
(From Scratch)"* by Sebastian Raschka, which I am currently reading
and studying. The code here represents my own attempts, notes, and
solutions to the exercises proposed in the book — written for learning
purposes as I progress through the chapters.

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
