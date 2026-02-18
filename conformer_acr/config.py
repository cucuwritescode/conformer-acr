"""
conformer_acr.config
~~~~~~~~~~~~~~~~~~~~

Global constants for audio processing and feature extraction.
Centralises all magic numbers so they can be imported once and
shared across every submodule.
"""

from typing import Final

# ── Audio ────────────────────────────────────────────────────────────
SR: Final[int] = 22050
"""Sample rate in Hz used for all audio loading."""

HOP_LENGTH: Final[int] = 512
"""Hop length in samples for STFT / CQT frames."""

# ── CQT / Chroma ────────────────────────────────────────────────────
N_CQT_BINS: Final[int] = 84
"""Number of CQT frequency bins (7 octaves × 12 semitones)."""

N_CHROMA: Final[int] = 12
"""Number of chroma bins (one per pitch class)."""

N_OCTAVES: Final[int] = 7
"""Number of octaves spanned by the CQT."""

FMIN: Final[float] = 32.70
"""Lowest CQT frequency in Hz (C1 ≈ 32.70 Hz)."""

# ── Model ────────────────────────────────────────────────────────────
NUM_ROOTS: Final[int] = 12
"""Number of root pitch classes."""

NUM_QUALITIES: Final[int] = 12
"""Number of chord quality classes (maj, min, 7, maj7, min7, …)."""

NUM_BASS: Final[int] = 13
"""Number of bass classes (12 pitch classes + no-bass)."""
