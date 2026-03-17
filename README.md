# Conformer-Based Long-Tail Automatic Chord Recognition
<p align="center">
  <img width="300" height="200" alt="acr_with_Bede" src="https://github.com/user-attachments/assets/bdc8a0db-21ea-4e9d-9b2b-947a5cf23df0" />
</p>

<p align="center">
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white"></a>
    <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
</p>

## Overview & Architecture

This library implements a multi-headed Conformer network designed to solve the "long-tail" problem in Automatic Chord Recognition (ACR).

<p align="center">
  <img width="300" alt="Architecture Diagram" src="https://github.com/user-attachments/assets/2f7dbddb-84fc-4edd-8163-e88447432c65" />
  <br>
  <em>Figure 1: The multi-headed Conformer architecture branching into Root, Bass, and Quality predictions.</em>
</p>

To counteract this, `conformer-acr` makes use of:
* **The Conformer Backbone:** Combines Convolutional Neural Networks (CNNs) to capture local acoustic texture/timbre with Transformers (self-attention) to maintain global harmonic context.
* **Structured Multi-Task Heads:** Instead of predicting a single monolithic chord string, the network branches into three distinct classification heads: **Root**, **Bass**, and **Quality**. This explicitly forces the model to understand inversions without causing a combinatorial explosion in the target vocabulary.
* **Synthetic Pre-Training (Harmonic Prior):** Because Conformers are memory and data-hungry, the model is pre-trained on perfectly annotated synthetic multitracks (the AAM dataset) using the Bede NVLink GPU cluster. This establishes a mathematically pure "harmonic prior" before the model is fine-tuned on noisy, real-world acoustic audio.

## Install

```bash
# editable install (for development)
pip install -e .

# with dev tools (pytest, etc)
pip install -e ".[dev]"


To counteract this, `conformer-acr` makes use of:
* **The Conformer Backbone:** Combines Convolutional Neural Networks (CNNs) to capture local acoustic texture/timbre with Transformers (self-attention) to maintain global harmonic context.
* **Structured Multi-Task Heads:** Instead of predicting a single monolithic chord string, the network branches into three distinct classification heads: **Root**, **Bass**, and **Quality**. This explicitly forces the model to understand inversions without causing a combinatorial explosion in the target vocabulary.
* **Synthetic Pre-Training (Harmonic Prior):** Because Conformers are memory and data-hungry, the model is pre-trained on perfectly annotated synthetic multitracks (the AAM dataset) using the Bede NVLink GPU cluster. This establishes a mathematically pure "harmonic prior" before the model is fine-tuned on noisy, real-world acoustic audio.

## Install

```bash
# editable install (for development)
pip install -e .

# with dev tools (pytest, etc)
pip install -e ".[dev]"
<img width="621" height="724" alt="Screenshot 2026-03-13 at 16 27 13" src="https://github.com/user-attachments/assets/2f7dbddb-84fc-4edd-8163-e88447432c65" />



## Install

```bash
#editable install (for development)
pip install -e .

#with dev tools (pytest, etc)
pip install -e ".[dev]"
```

## Quick Start

```python
import conformer_acr as acr

#feature extraction
cqt = acr.preprocess_audio("song.mp3")

#inference (requires a trained checkpoint)
chords = acr.predict("song.mp3", checkpoint_path="model.pt")

#model
model = acr.ConformerACR(d_model=256, n_heads=4, n_layers=4)

#chord vocabulary
idx   = acr.chord_to_index("C:maj")   # → 0
label = acr.index_to_chord(0)          # → 'C:maj'
```

## Library Structure

```
conformer_acr/
├── __init__.py          #flat public API
├── config.py            #constants (SR, CQT bins, hop length)
├── core.py              #high-level inference pipeline
├── models/
│   └── conformer.py     #ConformerACR (encoder + 3 heads)
├── data/
│   ├── dataset.py       #AAM & Isophonics Dataset classes
│   └── preprocess.py    #audio loading & CQT extraction
├── theory/
│   └── vocabulary.py    #chord ↔ integer mappings
├── training/
│   ├── trainer.py       #training loop
│   └── losses.py        #focal Loss
└── utils/
    └── distributed.py   #Bede/DDP helpers
```

## `lit_review/`

The `lit_review/` directory contains standalone research scripts and datasets used during the literature review phase. It is **not** part of the library.
