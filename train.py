#!/usr/bin/env python3
"""
train.py - SLURM-compatible training script for ConformerACR.

Automatically scans dataset labels to build vocabulary before training.

Usage:
    python train.py --data-dir /path/to/data --index-file /path/to/index.csv
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from conformer_acr.models.conformer import ConformerACR
from conformer_acr.training.trainer import Trainer
from conformer_acr.training.losses import FocalLoss
from conformer_acr.data.dataset import AAMDataset, pad_collate_fn, load_labels


# ============================================================================
#vocabulary builder
# ============================================================================

#standard chord roots (pitch classes)
ROOT_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N']
ROOT_TO_IDX = {name: i for i, name in enumerate(ROOT_NAMES)}

#also handle flats -> convert to sharps
FLAT_TO_SHARP = {
    'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#',
    'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B'
}


class VocabMapper:
    """
    Dynamically built vocabulary mapper for chord labels.

    Parses chord strings like "C:maj", "A:min", "G:7", "N" (no chord)
    and converts them to (root_idx, quality_idx, bass_idx) tuples.
    """

    def __init__(self, qualities: List[str]):
        """
        Parameters
        ----------
        qualities : List[str]
            List of unique quality strings found in the dataset.
        """
        self.root_to_idx = ROOT_TO_IDX
        self.qualities = sorted(set(qualities))
        #add 'N' for no chord if not present
        if 'N' not in self.qualities:
            self.qualities.append('N')
        self.quality_to_idx = {q: i for i, q in enumerate(self.qualities)}

        #bass uses same pitch classes as root
        self.bass_to_idx = ROOT_TO_IDX

        print(f"VocabMapper initialized:", flush=True)
        print(f"  Roots: {len(self.root_to_idx)} classes", flush=True)
        print(f"  Qualities: {len(self.quality_to_idx)} classes -> {self.qualities}", flush=True)
        print(f"  Bass: {len(self.bass_to_idx)} classes", flush=True)

    def parse_chord(self, chord_str: str) -> Tuple[int, int, int]:
        """
        Parse a chord string into (root_idx, quality_idx, bass_idx).

        Handles formats like:
            - "C:maj" -> (0, maj_idx, 0)
            - "A:min" -> (9, min_idx, 9)
            - "G:7/B" -> (7, 7_idx, 11)  # slash chord
            - "N" -> (12, N_idx, 12)     # no chord
        """
        chord_str = chord_str.strip()

        #handle "N" (no chord)
        if chord_str == 'N' or chord_str.lower() == 'n' or chord_str == '':
            n_idx = self.root_to_idx['N']
            return (n_idx, self.quality_to_idx.get('N', 0), n_idx)

        #check for slash chord (e.g., "C:maj/E")
        bass_note = None
        if '/' in chord_str:
            chord_str, bass_note = chord_str.rsplit('/', 1)

        #parse root and quality (handles C:maj or Cmaj notation)
        match = re.match(r'([A-G][#b]?):?(.*)', chord_str)
        if match:
            root_str = match.group(1)
            quality_str = match.group(2) if match.group(2) else 'maj'
        else:
            root_str = chord_str
            quality_str = 'maj'

        #normalise root (handle flats)
        root_str = FLAT_TO_SHARP.get(root_str, root_str)
        root_idx = self.root_to_idx.get(root_str, self.root_to_idx['N'])

        #get quality index
        quality_idx = self.quality_to_idx.get(quality_str, 0)

        #parse bass note (defaults to root if no slash)
        if bass_note:
            bass_note = FLAT_TO_SHARP.get(bass_note, bass_note)
            bass_idx = self.bass_to_idx.get(bass_note, root_idx)
        else:
            bass_idx = root_idx

        return (root_idx, quality_idx, bass_idx)

    @property
    def num_roots(self) -> int:
        return len(self.root_to_idx)

    @property
    def num_qualities(self) -> int:
        return len(self.quality_to_idx)

    @property
    def num_bass(self) -> int:
        return len(self.bass_to_idx)


def scan_dataset_for_vocab(
    index_file: str,
    data_dir: str,
    val_index_file: Optional[str] = None,
) -> VocabMapper:
    """
    Scan all label files in the dataset to discover unique chord qualities.

    Parameters
    ----------
    index_file : str
        Path to training index CSV.
    data_dir : str
        Root directory containing label files.
    val_index_file : str, optional
        Path to validation index CSV (also scanned).

    Returns
    -------
    VocabMapper
        Initialized vocabulary mapper with discovered qualities.
    """
    print("Scanning dataset for vocabulary...", flush=True)

    all_qualities = set()
    all_chords = set()

    #gather all index files to scan
    index_files = [index_file]
    if val_index_file:
        index_files.append(val_index_file)

    for idx_file in index_files:
        metadata = pd.read_csv(idx_file)
        print(f"  Scanning {idx_file} ({len(metadata)} files)...", flush=True)

        for _, row in metadata.iterrows():
            label_path = os.path.join(data_dir, row["label_file"])
            if os.path.exists(label_path):
                label_df = load_labels(label_path)
                if 'chord' in label_df.columns:
                    for chord in label_df['chord'].unique():
                        chord = str(chord).strip()
                        all_chords.add(chord)

                        #extract quality from chord string
                        if chord == 'N' or chord.lower() == 'n' or chord == '':
                            all_qualities.add('N')
                        else:
                            #handle both C:maj and Cmaj notation
                            chord_part = chord.split('/')[0] if '/' in chord else chord
                            match = re.match(r'([A-G][#b]?):?(.*)', chord_part)
                            if match and match.group(2):
                                all_qualities.add(match.group(2))
                            else:
                                all_qualities.add('maj')

    print(f"  Found {len(all_chords)} unique chord labels", flush=True)
    print(f"  Found {len(all_qualities)} unique qualities: {sorted(all_qualities)}", flush=True)

    return VocabMapper(list(all_qualities))


# ============================================================================
#main training script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ConformerACR model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing audio and label files")
    parser.add_argument("--index-file", type=str, required=True,
                        help="CSV file with audio_file and label_file columns")
    parser.add_argument("--val-index-file", type=str, default=None,
                        help="CSV file for validation set (optional)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # ========================================================================
    #device setup
    # ========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA version: {torch.version.cuda}", flush=True)

    # ========================================================================
    #vocabulary discovery
    # ========================================================================
    vocab_mapper = scan_dataset_for_vocab(
        index_file=args.index_file,
        data_dir=args.data_dir,
        val_index_file=args.val_index_file,
    )

    # ========================================================================
    #model setup
    # ========================================================================
    #update model dimensions based on discovered vocabulary
    model = ConformerACR(
        #these should match your config.py or be overridden here
        # num_roots=vocab_mapper.num_roots,
        # num_qualities=vocab_mapper.num_qualities,
        # num_bass=vocab_mapper.num_bass,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = FocalLoss()

    #resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}", flush=True)

    # ========================================================================
    #dataset setup
    # ========================================================================
    train_dataset = AAMDataset(
        index_file=args.index_file,
        audio_dir=args.data_dir,
        vocab_mapper=vocab_mapper,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )
    print(f"Training samples: {len(train_dataset)}", flush=True)

    val_loader = None
    if args.val_index_file:
        val_dataset = AAMDataset(
            index_file=args.val_index_file,
            audio_dir=args.data_dir,
            vocab_mapper=vocab_mapper,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
        )
        print(f"Validation samples: {len(val_dataset)}", flush=True)

    # ========================================================================
    #training
    # ========================================================================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"Starting training for {args.epochs} epochs...", flush=True)
    print("=" * 60, flush=True)

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
    )

    print("=" * 60, flush=True)
    print("Training complete!", flush=True)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}", flush=True)
    if val_loader:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
