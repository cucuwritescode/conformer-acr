"""
conformer_acr
~~~~~~~~~~~~~

Conformer-based Automatic Chord Recognition.

Quick-start::

    import conformer_acr as acr

    # Feature extraction
    cqt = acr.preprocess_audio("song.mp3")

    # Full pipeline (requires a trained checkpoint)
    chords = acr.predict("song.mp3", checkpoint_path="model.pt")

    # Model instantiation
    model = acr.ConformerACR(d_model=256, n_heads=4, n_layers=4)

    # Chord vocabulary
    idx = acr.chord_to_index("C:maj")
    label = acr.index_to_chord(0)

Subpackages
-----------
models    Neural network architectures (Conformer + heads).
data      Dataset loaders and audio preprocessing.
theory    Musical-domain knowledge (chord ↔ integer mappings).
training  Training loop and loss functions.
utils     Distributed training helpers (Bede / DDP).
"""

from __future__ import annotations

__version__: str = "0.1.0"

# ── Core pipeline ────────────────────────────────────────────────────
from conformer_acr.core import predict, preprocess_audio

# ── Model ────────────────────────────────────────────────────────────
from conformer_acr.models.conformer import ConformerACR

# ── Data ─────────────────────────────────────────────────────────────
from conformer_acr.data.dataset import AAMDataset, IsophonicsDataset
from conformer_acr.data.preprocess import compute_chroma_cqt, compute_cqt, load_audio

# ── Theory ───────────────────────────────────────────────────────────
from conformer_acr.theory.vocabulary import (
    CHORD_LABELS,
    ROOT_LABELS,
    chord_to_index,
    index_to_chord,
    reduce_chord,
)

# ── Training ─────────────────────────────────────────────────────────
from conformer_acr.training.losses import FocalLoss
from conformer_acr.training.trainer import Trainer

# ── Config (re-export constants for convenience) ─────────────────────
from conformer_acr.config import HOP_LENGTH, N_CQT_BINS, SR

__all__: list[str] = [
    # pipeline
    "predict",
    "preprocess_audio",
    # model
    "ConformerACR",
    # data
    "AAMDataset",
    "IsophonicsDataset",
    "load_audio",
    "compute_cqt",
    "compute_chroma_cqt",
    # theory
    "CHORD_LABELS",
    "ROOT_LABELS",
    "chord_to_index",
    "index_to_chord",
    "reduce_chord",
    # training
    "FocalLoss",
    "Trainer",
    # config
    "SR",
    "HOP_LENGTH",
    "N_CQT_BINS",
]
