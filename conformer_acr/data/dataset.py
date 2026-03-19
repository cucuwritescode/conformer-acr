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
import re
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from conformer_acr.data.preprocess import extract_cqt


def parse_arff(filepath: str) -> pd.DataFrame:
    """
    Parse an ARFF file into a pandas DataFrame.

    Expects an ARFF file with attributes for start time, end time, and chord label.
    Common attribute names: start, end, chord, label, start_time, end_time, etc.

    Parameters
    ----------
    filepath : str
        Path to the .arff file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: start_time, end_time, chord
    """
    attributes = []
    data_rows = []
    in_data_section = False

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()

            #skip empty lines and comments
            if not line or line.startswith('%'):
                continue

            #parse attribute declarations
            if line.lower().startswith('@attribute'):
                #extract attribute name (handles quoted names)
                match = re.match(r"@attribute\s+['\"]?(\w+)['\"]?\s+", line, re.IGNORECASE)
                if match:
                    attributes.append(match.group(1).lower())
                continue

            #check for data section start
            if line.lower().startswith('@data'):
                in_data_section = True
                continue

            #parse data rows
            if in_data_section and line:
                #handle quoted strings in ARFF
                #split by comma, but respect quotes
                values = []
                current = ""
                in_quotes = False
                quote_char = None

                for char in line:
                    if char in ["'", '"'] and not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char and in_quotes:
                        in_quotes = False
                        quote_char = None
                    elif char == ',' and not in_quotes:
                        values.append(current.strip().strip("'\""))
                        current = ""
                    else:
                        current += char

                if current:
                    values.append(current.strip().strip("'\""))

                if values:
                    data_rows.append(values)

    #create DataFrame
    if not data_rows:
        return pd.DataFrame(columns=['start_time', 'end_time', 'chord'])

    df = pd.DataFrame(data_rows, columns=attributes[:len(data_rows[0])] if attributes else None)

    #normalise column names to expected format
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower() if isinstance(col, str) else str(col)
        if col_lower in ['start', 'start_time', 'starttime', 'onset']:
            column_mapping[col] = 'start_time'
        elif col_lower in ['end', 'end_time', 'endtime', 'offset']:
            column_mapping[col] = 'end_time'
        elif col_lower in ['chord', 'label', 'chordlabel', 'chord_label']:
            column_mapping[col] = 'chord'

    df = df.rename(columns=column_mapping)

    #make sure required columns exist
    if 'start_time' not in df.columns or 'end_time' not in df.columns:
        #try positional: assume first two columns are start/end
        cols = list(df.columns)
        if len(cols) >= 2:
            df = df.rename(columns={cols[0]: 'start_time', cols[1]: 'end_time'})
        if len(cols) >= 3 and 'chord' not in df.columns:
            df = df.rename(columns={cols[2]: 'chord'})

    #convert time columns to float
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce').fillna(0)
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce').fillna(0)

    return df


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
