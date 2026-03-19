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

        #extract cqt spectrogram, (time, 252)
        cqt_tensor = extract_cqt(audio_path)
        time_frames = cqt_tensor.shape[0]

        #load labels
        #label_file is a CSV with 'start_time', 'end_time', 'chord'
        label_df = pd.read_csv(os.path.join(self.audio_dir, row["label_file"]))

        #initialise label tensors
        root_labels = torch.zeros(time_frames, dtype=torch.long)
        bass_labels = torch.zeros(time_frames, dtype=torch.long)
        qual_labels = torch.zeros(time_frames, dtype=torch.long)

        #align labels to CQT frames using timestamps
        #cqt frame rate: SR / HOP_LENGTH frames per second
        from conformer_acr.config import SR, HOP_LENGTH
        frame_duration = HOP_LENGTH / SR  #seconds per frame

        for _, label_row in label_df.iterrows():
            start_time = float(label_row['start_time'])
            end_time = float(label_row['end_time'])
            chord_str = str(label_row['chord'])

            #convert timestamps to frame indices
            start_frame = int(start_time / frame_duration)
            end_frame = int(end_time / frame_duration)

            #clamp to valid range
            start_frame = max(0, min(start_frame, time_frames - 1))
            end_frame = max(0, min(end_frame, time_frames))

            #parse chord using vocab_mapper
            if self.vocab_mapper is not None:
                root_idx, qual_idx, bass_idx = self.vocab_mapper.parse_chord(chord_str)
            else:
                root_idx, qual_idx, bass_idx = 0, 0, 0

            #assign labels to frames
            root_labels[start_frame:end_frame] = root_idx
            qual_labels[start_frame:end_frame] = qual_idx
            bass_labels[start_frame:end_frame] = bass_idx

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
