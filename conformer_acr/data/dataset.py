"""
conformer_acr.data.dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch Dataset for the AAM dataset and a collate function that pads
variable-length CQT sequences so PyTorch can assemble batches.

The magic: ``padding_value=-100`` tells ``CrossEntropyLoss`` to ignore
padded frames, so the model is never penalised for predicting during
silence at the tail of a shorter clip.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from conformer_acr.data.preprocess import extract_cqt


class AAMDataset(Dataset):  #type: ignore[type-arg]
    """PyTorch Dataset for the Artificial Audio Multitracks (AAM) dataset.

    Parameters
    ----------
    index_file : str
        Path to a CSV containing ``audio_file`` and ``label_file`` columns.
    audio_dir : str
        Root directory containing the generated ``.wav`` files.
    vocab_mapper
        Object that converts string labels (e.g. ``"C:maj"``) to
        ``(root, bass, quality)`` integer indices.
    """

    def __init__(self, index_file: str, audio_dir: str, vocab_mapper) -> None:
        self.audio_dir = audio_dir
        self.metadata = pd.read_csv(index_file)
        self.vocab_mapper = vocab_mapper

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["audio_file"])

        #extract CQT Spectrogram, (time, 252)
        cqt_tensor = extract_cqt(audio_path)
        time_frames = cqt_tensor.shape[0]

        #load and align labels
        #label_file is a CSV with 'start_time', 'end_time', 'chord'
        #in production you'd sample the label at each CQT time frame
        label_df = pd.read_csv(os.path.join(self.audio_dir, row["label_file"]))

        root_labels = torch.zeros(time_frames, dtype=torch.long)
        bass_labels = torch.zeros(time_frames, dtype=torch.long)
        qual_labels = torch.zeros(time_frames, dtype=torch.long)

        # TODO: Replace with exact timestamp → frame alignment logic
        #root_labels[:], bass_labels[:], qual_labels[:] = self.vocab_mapper(label_df)

        return cqt_tensor, root_labels, bass_labels, qual_labels


def pad_collate_fn(
    batch: List[Tuple],
) -> Dict[str, torch.Tensor]:
    """Pad variable-length CQT sequences and labels to max length in batch.

    Uses ``padding_value=-100`` for labels so that
    :class:`torch.nn.CrossEntropyLoss` (which ignores index ``-100`` by
    default) skips the padded frames automatically.

    Returns
    -------
    dict
        ``cqt``      — ``(Batch, Max_Time, 252)``
        ``lengths``  — ``(Batch,)`` original sequence lengths
        ``root``     — ``(Batch, Max_Time)``
        ``bass``     — ``(Batch, Max_Time)``
        ``qual``     — ``(Batch, Max_Time)``
    """
    cqts, roots, basses, quals = zip(*batch)

    #sequence lengths for conformer attention masking
    lengths = torch.tensor([cqt.shape[0] for cqt in cqts], dtype=torch.long)

    #pad sequences: (Batch, Max_Time, Features) / (Batch, Max_Time)
    padded_cqts = torch.nn.utils.rnn.pad_sequence(cqts, batch_first=True)
    padded_roots = torch.nn.utils.rnn.pad_sequence(
        roots, batch_first=True, padding_value=-100
    )
    padded_basses = torch.nn.utils.rnn.pad_sequence(
        basses, batch_first=True, padding_value=-100
    )
    padded_quals = torch.nn.utils.rnn.pad_sequence(
        quals, batch_first=True, padding_value=-100
    )

    return {
        "cqt": padded_cqts,
        "lengths": lengths,
        "root": padded_roots,
        "bass": padded_basses,
        "qual": padded_quals,
    }
