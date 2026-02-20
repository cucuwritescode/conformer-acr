"""
conformer_acr.data.preprocess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Audio loading and CQT feature extraction for the Conformer ACR model.

Handles the heavy DSP lifting: loads raw audio, computes Constant-Q Transform
spectrograms with 252 frequency bins (36 per octave × 7 octaves), and formats
the output as PyTorch tensors ready for the Conformer.
"""

from __future__ import annotations

import numpy as np
import librosa
import torch
from typing import Tuple

from conformer_acr.config import (
    BINS_PER_OCTAVE,
    FMIN,
    HOP_LENGTH,
    N_CQT_BINS,
    SR,
)

#re-export config constants for convenience / backward compat
SAMPLE_RATE = SR
N_BINS = N_CQT_BINS


def extract_cqt(audio_path: str) -> torch.Tensor:
    """
    Load an audio file and compute the Constant-Q Transform (CQT) spectrogram.

    This is the primary feature-extraction entry point for training and
    inference.  It produces a log-scaled magnitude CQT with 252 frequency bins
    (36 bins per octave across 7 octaves), matching the ``input_dim=252``
    expected by :class:`~conformer_acr.models.ConformerACR`.

    Parameters
    ----------
    audio_path : str
        Path to the ``.wav`` or ``.mp3`` file.

    Returns
    -------
    torch.Tensor
        CQT magnitude spectrogram of shape ``(Time, Freq_Bins)``.

    Notes
    -----
    * 36 bins per octave gives 3 bins per semitone — mandatory for chord
      recognition so the model can distinguish slight detuning from actual
      harmonic shifts.
    * ``amplitude_to_db`` compresses dynamic range, preventing loud transients
      from masking quieter harmonic content (e.g. bass notes under drum hits).
    """
    # load audio
    # librosa automatically resamples to SAMPLE_RATE and converts to mono
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # compute CQT
    # minimum frequency of C1 (≈ 32.7 Hz)
    C = librosa.cqt(
        y,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=librosa.note_to_hz('C1'),
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )

    # magnitude and log scaling
    # neural networks prefer log-scaled inputs rather than raw amplitudes
    C_mag = np.abs(C)
    C_db = librosa.amplitude_to_db(C_mag, ref=np.max)

    # format for pytorch
    # PyTorch sequence models expect (Batch, Time, Features).
    # librosa outputs (Features, Time), so transpose here.
    cqt_tensor = torch.from_numpy(C_db).T

    return cqt_tensor


def get_time_frames(audio_length_samples: int) -> np.ndarray:
    """Map CQT frame indices back to actual timestamps (seconds).

    Useful during evaluation to align predicted chords with ground-truth
    annotations that are specified in seconds.

    Parameters
    ----------
    audio_length_samples : int
        Length of the original audio signal in samples.

    Returns
    -------
    np.ndarray
        Array of timestamps (in seconds) for each CQT frame.
    """
    return librosa.frames_to_time(
        np.arange(audio_length_samples // HOP_LENGTH + 1),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
