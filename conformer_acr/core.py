"""
conformer_acr.core
~~~~~~~~~~~~~~~~~~

High-level inference pipeline — the "glue" that connects
preprocessing, model, and vocabulary decoding into one-liner calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from conformer_acr.data.preprocess import extract_cqt
from conformer_acr.models.conformer import ConformerACR
from conformer_acr.theory.vocabulary import index_to_chord


def preprocess_audio(
    path: str | Path,
) -> torch.Tensor:
    """Load audio and extract CQT features ready for the model.

    Parameters
    ----------
    path : str | Path
        Path to an audio file.

    Returns
    -------
    torch.Tensor, shape ``(Time, Freq_Bins)``
        Log-scaled CQT magnitude spectrogram.
    """
    return extract_cqt(str(path))


def predict(
    audio_path: str | Path,
    model: ConformerACR | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> List[str]:
    """Run end-to-end chord prediction on an audio file.

    Parameters
    ----------
    audio_path : str | Path
        Path to the input audio file.
    model : ConformerACR, optional
        A pre-loaded model instance. If *None*, a fresh model is
        instantiated and weights are loaded from *checkpoint_path*.
    checkpoint_path : str | Path, optional
        Path to a ``.pt`` checkpoint (required when *model* is *None*).
    device : str | torch.device
        Target device for inference.

    Returns
    -------
    list[str]
        One chord label per CQT frame (e.g. ``['C:maj', 'C:maj', 'A:min', …]``).
    """
    if model is None:
        model = ConformerACR()
        if checkpoint_path is not None:
            state = torch.load(str(checkpoint_path), map_location=device)
            model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()

    # extract_cqt returns (Time, Freq_Bins) as a torch.Tensor
    cqt = preprocess_audio(audio_path)

    # Add batch dimension: (Time, Freq_Bins) → (1, Time, Freq_Bins)
    x = cqt.unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(x)

    root_ids: np.ndarray = out["root"].argmax(dim=-1).squeeze(0).cpu().numpy()
    # TODO: combine root + quality + bass heads into full chord labels.
    #       For now, return root-only labels.
    chords: List[str] = [index_to_chord(int(idx)) for idx in root_ids]
    return chords

