"""
conformer_acr.data.dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch Dataset classes for the two supported annotation formats:

* **AAM** — Audio Analysis Metadata (JAMS-like)
* **Isophonics** — Chris Harte's Beatles / Queen / Zweieck annotations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from conformer_acr.data.preprocess import compute_cqt, load_audio
from conformer_acr.theory.vocabulary import chord_to_index


class AAMDataset(Dataset):  # type: ignore[type-arg]
    """Dataset backed by Audio Analysis Metadata (AAM / JAMS) files.

    Parameters
    ----------
    audio_dir : str | Path
        Directory containing audio files.
    annotation_dir : str | Path
        Directory containing ``.jams`` annotation files.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        annotation_dir: str | Path,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.annotation_dir = Path(annotation_dir)
        self._index: list[tuple[Path, Path]] = []
        # TODO: Build file-pair index by matching stems.

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        audio_path, ann_path = self._index[idx]
        y, sr = load_audio(audio_path)
        cqt = compute_cqt(y, sr)
        # TODO: Parse JAMS annotations → target tensors.
        target = np.zeros(cqt.shape[1], dtype=np.int64)
        return {
            "cqt": torch.from_numpy(cqt),
            "target": torch.from_numpy(target),
        }


class IsophonicsDataset(Dataset):  # type: ignore[type-arg]
    """Dataset for the Isophonics chord annotation format (``.lab``).

    Parameters
    ----------
    audio_dir : str | Path
        Directory containing audio files.
    lab_dir : str | Path
        Directory containing ``.lab`` annotation files.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        lab_dir: str | Path,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.lab_dir = Path(lab_dir)
        self._index: list[tuple[Path, Path]] = []
        # TODO: Build file-pair index by matching stems.

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        audio_path, lab_path = self._index[idx]
        y, sr = load_audio(audio_path)
        cqt = compute_cqt(y, sr)
        # TODO: Parse .lab annotations → target tensors.
        target = np.zeros(cqt.shape[1], dtype=np.int64)
        return {
            "cqt": torch.from_numpy(cqt),
            "target": torch.from_numpy(target),
        }
