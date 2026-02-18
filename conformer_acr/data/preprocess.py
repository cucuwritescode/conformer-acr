"""
conformer_acr.data.preprocess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

audio loading and CQT feature extraction.
thin, typed wrappers around :mod:`librosa` so the rest of the
library never calls librosa directly.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from conformer_acr.config import FMIN, HOP_LENGTH, N_CQT_BINS, N_OCTAVES, SR


def load_audio(
    path: str | Path,
    sr: int = SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """load an audio file and resample to *sr* Hz.

    parameters
    ----------
    path : str | Path
        path to any format supported by ``soundfile`` / ``audioread``.
    sr : int
        target sample rate (default: :data:`conformer_acr.config.SR`).
    mono : bool
        down-mix to mono if *True*.

    Returns
    -------
    y : np.ndarray, shape ``(n_samples,)``
        Audio time-series.
    sr : int
        Actual sample rate (always equals the requested *sr*).
    """
    y, sr_out = librosa.load(str(path), sr=sr, mono=mono)
    return y, sr_out


def compute_cqt(
    y: np.ndarray,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    n_bins: int = N_CQT_BINS,
    fmin: float = FMIN,
) -> np.ndarray:
    """compute the Constant-Q Transform magnitude spectrogram.

    parameters
    ----------
    y : np.ndarray
        audio time-series (mono).
    sr : int
        sample rate.
    hop_length : int
        hop between successive CQT frames.
    n_bins : int
        total number of CQT frequency bins.
    fmin : float
        minimum frequency (Hz).

    returns
    -------
    np.ndarray, shape ``(n_bins, n_frames)``
        magnitude CQT spectrogram.
    """
    cqt: np.ndarray = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            fmin=fmin,
        )
    )
    return cqt


def compute_chroma_cqt(
    y: np.ndarray,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    n_octaves: int = N_OCTAVES,
) -> np.ndarray:
    """compute a CQT-based chromagram.

    parameters
    ----------
    y : np.ndarray
        audio time-series (mono).
    sr : int
        sample rate.
    hop_length : int
        hop between successive frames.
    n_octaves : int
        number of octaves spanned.

    returns
    -------
    np.ndarray, shape ``(12, n_frames)``
        chromagram normalised per-frame.
    """
    chroma: np.ndarray = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_octaves=n_octaves,
    )
    return chroma
