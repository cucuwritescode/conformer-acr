#!/usr/bin/env python3
"""
prep_dataset.py - Generate index.csv for training.

Scans a data directory for matching .flac audio and .arff label file pairs,
then generates an index.csv that train.py needs.

Expected naming convention:
    Audio: XXXX_mix.flac  (e.g., 0001_mix.flac)
    Labels: XXXX_segments.arff  (e.g., 0001_segments.arff)

Usage:
    python prep_dataset.py --data-dir /path/to/data --output index.csv
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd


def find_audio_label_pairs(
    data_dir: str,
    audio_pattern: str = r"(\d+)_mix\.flac$",
    label_pattern: str = r"(\d+)_segments\.arff$",
) -> list[tuple[str, str]]:
    """
    Find matching audio and label file pairs in a directory.

    Parameters
    ----------
    data_dir : str
        Root directory to scan (recursively).
    audio_pattern : str
        Regex pattern to match audio files. Must have a capture group for the ID.
    label_pattern : str
        Regex pattern to match label files. Must have a capture group for the ID.

    Returns
    -------
    list[tuple[str, str]]
        List of (audio_file, label_file) tuples with paths relative to data_dir.
    """
    audio_regex = re.compile(audio_pattern)
    label_regex = re.compile(label_pattern)

    #collect all audio and label files with their IDs
    audio_files: dict[str, str] = {}  # id -> relative path
    label_files: dict[str, str] = {}  # id -> relative path

    data_path = Path(data_dir)

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, data_dir)

            #check if it's an audio file
            audio_match = audio_regex.search(filename)
            if audio_match:
                file_id = audio_match.group(1)
                audio_files[file_id] = rel_path
                continue

            #check if it's a label file
            label_match = label_regex.search(filename)
            if label_match:
                file_id = label_match.group(1)
                label_files[file_id] = rel_path

    #match audio and label files by ID
    pairs = []
    matched_ids = set(audio_files.keys()) & set(label_files.keys())

    for file_id in sorted(matched_ids):
        pairs.append((audio_files[file_id], label_files[file_id]))

    #report unmatched files
    unmatched_audio = set(audio_files.keys()) - matched_ids
    unmatched_labels = set(label_files.keys()) - matched_ids

    if unmatched_audio:
        print(f"Warning: {len(unmatched_audio)} audio files without matching labels:")
        for fid in sorted(unmatched_audio)[:5]:
            print(f"  - {audio_files[fid]}")
        if len(unmatched_audio) > 5:
            print(f"  ... and {len(unmatched_audio) - 5} more")

    if unmatched_labels:
        print(f"Warning: {len(unmatched_labels)} label files without matching audio:")
        for fid in sorted(unmatched_labels)[:5]:
            print(f"  - {label_files[fid]}")
        if len(unmatched_labels) > 5:
            print(f"  ... and {len(unmatched_labels) - 5} more")

    return pairs


def create_index_csv(
    pairs: list[tuple[str, str]],
    output_path: str,
    train_ratio: float = 0.8,
    create_splits: bool = True,
) -> None:
    """
    Create index CSV file(s) from audio-label pairs.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        List of (audio_file, label_file) tuples.
    output_path : str
        Path to output CSV file.
    train_ratio : float
        Ratio of data to use for training (rest goes to validation).
    create_splits : bool
        If True, create separate train_index.csv and val_index.csv files.
    """
    df = pd.DataFrame(pairs, columns=["audio_file", "label_file"])

    #save full index
    df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(df)} entries")

    if create_splits and len(df) > 1:
        #shuffle and split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df_shuffled) * train_ratio)

        train_df = df_shuffled[:split_idx]
        val_df = df_shuffled[split_idx:]

        #derive output paths
        output_dir = os.path.dirname(output_path) or "."
        train_path = os.path.join(output_dir, "train_index.csv")
        val_path = os.path.join(output_dir, "val_index.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        print(f"Created {train_path} with {len(train_df)} entries (training)")
        print(f"Created {val_path} with {len(val_df)} entries (validation)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate index.csv for ConformerACR training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing audio and label files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="index.csv",
        help="Output CSV file path (default: index.csv)",
    )
    parser.add_argument(
        "--audio-pattern",
        type=str,
        default=r"(\d+)_mix\.flac$",
        help="Regex pattern for audio files (default: (\\d+)_mix\\.flac$)",
    )
    parser.add_argument(
        "--label-pattern",
        type=str,
        default=r"(\d+)_segments\.arff$",
        help="Regex pattern for label files (default: (\\d+)_segments\\.arff$)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training split (default: 0.8)",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Only create single index.csv, no train/val splits",
    )
    args = parser.parse_args()

    print(f"Scanning {args.data_dir} for audio-label pairs...")
    print(f"  Audio pattern: {args.audio_pattern}")
    print(f"  Label pattern: {args.label_pattern}")
    print()

    pairs = find_audio_label_pairs(
        data_dir=args.data_dir,
        audio_pattern=args.audio_pattern,
        label_pattern=args.label_pattern,
    )

    if not pairs:
        print("Error: No matching audio-label pairs found!")
        print("Check your --data-dir and patterns.")
        return

    print(f"\nFound {len(pairs)} matching audio-label pairs")
    print()

    create_index_csv(
        pairs=pairs,
        output_path=args.output,
        train_ratio=args.train_ratio,
        create_splits=not args.no_splits,
    )

    print("\nDone! You can now run training with:")
    if not args.no_splits:
        print(f"  python train.py --data-dir {args.data_dir} \\")
        print(f"      --index-file train_index.csv \\")
        print(f"      --val-index-file val_index.csv")
    else:
        print(f"  python train.py --data-dir {args.data_dir} --index-file {args.output}")


if __name__ == "__main__":
    main()
