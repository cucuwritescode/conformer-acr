# conformer-acr

> Deep Learning Models for Automatic Chord Recognition in Polyphonic Audio — N8 CIR Bede Supercomputer studentship.

## Install

```bash
# Editable install (for development)
pip install -e .

# With dev tools (pytest, black, flake8)
pip install -e ".[dev]"
```

## Quick Start

```python
import conformer_acr as acr

# Feature extraction
cqt = acr.preprocess_audio("song.mp3")

# Inference (requires a trained checkpoint)
chords = acr.predict("song.mp3", checkpoint_path="model.pt")

# Model
model = acr.ConformerACR(d_model=256, n_heads=4, n_layers=4)

# Chord vocabulary
idx   = acr.chord_to_index("C:maj")   # → 0
label = acr.index_to_chord(0)          # → 'C:maj'
```

## Library Structure

```
conformer_acr/
├── __init__.py          # Flat public API
├── config.py            # Constants (SR, CQT bins, hop length)
├── core.py              # High-level inference pipeline
├── models/
│   └── conformer.py     # ConformerACR (encoder + 3 heads)
├── data/
│   ├── dataset.py       # AAM & Isophonics Dataset classes
│   └── preprocess.py    # Audio loading & CQT extraction
├── theory/
│   └── vocabulary.py    # Chord ↔ integer mappings
├── training/
│   ├── trainer.py       # Training loop
│   └── losses.py        # Focal Loss
└── utils/
    └── distributed.py   # Bede / DDP helpers
```

## `lit_review/`

The `lit_review/` directory contains standalone research scripts and datasets used during the literature review phase. It is **not** part of the library.
