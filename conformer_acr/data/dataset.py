"""
conformer_acr.data.dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch Dataset for the AAM dataset and a collate function that pads
variable-length CQT sequences so PyTorch can assemble batches.

The magic: ``padding_value=-100`` tells ``CrossEntropyLoss`` to ignore
padded frames, so the model is never penalised for predicting during
silence at the tail of a shorter clip.

Supports:
- Audio: .wav, .flac, .mp3 (anything librosa can load)
- Labels: .csv or .arff format
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from conformer_acr.data.preprocess import extract_cqt


def parse_arff(filepath: str) -> pd.DataFrame:
    """
    Parse a beatinfo ARFF file into a pandas DataFrame.

    Beatinfo files have 4 columns: start_time, bar_count, quarter_count, chord.
    End times are calculated dynamically from the next chord's start time.

    Parameters
    ----------
    filepath : str
        Path to the .arff file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: start_time, end_time, chord
    """
    data_rows = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            #skip empty lines, comments, and headers
            if not line or line.startswith('%') or line.startswith('@'):
                continue

            #if we made it here, it's a raw data row
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    start_time = float(parts[0])
                    #chord is always the last column
                    chord = parts[-1].strip().strip("'\"")
                    data_rows.append([start_time, chord])
                except ValueError:
                    pass

    if not data_rows:
        return pd.DataFrame(columns=['start_time', 'end_time', 'chord'])

    df = pd.DataFrame(data_rows, columns=['start_time', 'chord'])
    #calculate end_time dynamically
    df['end_time'] = df['start_time'].shift(-1)
    df['end_time'] = df['end_time'].fillna(df['start_time'].iloc[-1] + 10.0)

    return df[['start_time', 'end_time', 'chord']]


def load_labels(filepath: str) -> pd.DataFrame:
    """
    Load label file in either CSV or ARFF format.

    Parameters
    ----------
    filepath : str
        Path to label file (.csv or .arff)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: start_time, end_time, chord
    """
    if filepath.lower().endswith('.arff'):
        return parse_arff(filepath)
    else:
        return pd.read_csv(filepath)


class AAMDataset(Dataset):  #type: ignore[type-arg]
    """PyTorch Dataset for chord recognition.

    Supports:
    - Audio: .wav, .flac, .mp3 (anything librosa can load)
    - Labels: .csv or .arff format

    Parameters
    ----------
    index_file : str
        Path to a CSV containing ``audio_file`` and ``label_file`` columns.
    audio_dir : str
        Root directory containing audio and label files.
    vocab_mapper
        Object that converts string labels (e.g. ``"C:maj"``) to
        ``(root, quality, bass)`` integer indices.
    """

    def __init__(self, index_file: str, audio_dir: str, vocab_mapper, max_seq_len: int = 2048, random_crop: bool = True) -> None:
        self.audio_dir = audio_dir
        self.metadata = pd.read_csv(index_file)
        self.vocab_mapper = vocab_mapper
        self.max_seq_len = max_seq_len  #truncate long sequences to avoid OOM
        self.random_crop = random_crop  #False for val = deterministic center crop

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.audio_dir, row["audio_file"])

        #try to load pre-computed CQT tensor first (faster)
        #render_hybrid.py saves these as *_cqt.pt alongside *_mix.flac
        cqt_path = audio_path.replace('_mix.flac', '_cqt.pt').replace('.flac', '_cqt.pt')
        if os.path.exists(cqt_path):
            cqt_tensor = torch.load(cqt_path, map_location="cpu")
        else:
            #fallback: extract CQT on the fly
            cqt_tensor = extract_cqt(audio_path)
        time_frames = cqt_tensor.shape[0]

        #load labels (supports .csv or .arff)
        label_path = os.path.join(self.audio_dir, row["label_file"])
        label_df = load_labels(label_path)

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

        #replace N (no-chord) labels with -100 so loss ignores them
        #this fixes 93% class imbalance by not training on silent frames
        if self.vocab_mapper is not None:
            n_root = self.vocab_mapper.root_to_idx.get('N', -1)
            n_qual = self.vocab_mapper.quality_to_idx.get('N', -1)
            n_bass = self.vocab_mapper.bass_to_idx.get('N', -1)
            if n_root >= 0:
                root_labels[root_labels == n_root] = -100
            if n_qual >= 0:
                qual_labels[qual_labels == n_qual] = -100
            if n_bass >= 0:
                bass_labels[bass_labels == n_bass] = -100

        #crop long sequences to avoid OOM in attention
        if self.max_seq_len and time_frames > self.max_seq_len:
            if self.random_crop:
                import random
                start = random.randint(0, time_frames - self.max_seq_len)
            else:
                #center crop for validation (deterministic)
                start = (time_frames - self.max_seq_len) // 2
            end = start + self.max_seq_len
            cqt_tensor = cqt_tensor[start:end]
            root_labels = root_labels[start:end]
            bass_labels = bass_labels[start:end]
            qual_labels = qual_labels[start:end]

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
