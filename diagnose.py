#!/usr/bin/env python3
"""
diagnose.py - Debug why the model is collapsing to one class.
Run on cluster after training to understand what's happening.

Usage:
    python diagnose.py --checkpoint checkpoints/checkpoint_epoch0100.pt \
                       --data-dir /nobackup/projects/bdyrk27/slakh_workspace/slakh_audio \
                       --index-file val_index.csv
"""

import argparse
import json
import os
import glob
import torch
import torch.nn.functional as F
import numpy as np

from conformer_acr.models.conformer import ConformerACR
from conformer_acr.data.dataset import AAMDataset, pad_collate_fn
from torch.utils.data import DataLoader


def test_raw_cqt_files(data_dir, num_files=5):
    """check raw pre-computed CQT .pt files"""
    print("\n" + "="*60)
    print("TEST A: Raw CQT File Inspection")
    print("="*60)

    cqt_files = glob.glob(os.path.join(data_dir, "**/*_cqt.pt"), recursive=True)
    print(f"Found {len(cqt_files)} CQT files in {data_dir}\n")

    if len(cqt_files) == 0:
        print("WARNING: No pre-computed CQT files found!")
        print("The dataset might be extracting CQT on-the-fly from audio.")
        return

    for i, cqt_path in enumerate(cqt_files[:num_files]):
        try:
            cqt = torch.load(cqt_path, map_location='cpu')
            print(f"File {i+1}: {os.path.basename(cqt_path)}")
            print(f"  Shape: {cqt.shape}")
            print(f"  Dtype: {cqt.dtype}")
            print(f"  Min: {cqt.min().item():.4f}")
            print(f"  Max: {cqt.max().item():.4f}")
            print(f"  Mean: {cqt.mean().item():.4f}")
            print(f"  Std: {cqt.std().item():.4f}")

            if cqt.std().item() < 0.001:
                print("  *** BROKEN: Zero variance - constant values! ***")
            if cqt.min().item() == cqt.max().item():
                print("  *** BROKEN: All values identical! ***")
            print()
        except Exception as e:
            print(f"  ERROR loading {cqt_path}: {e}\n")


def test_audio_files(data_dir, index_file, num_files=3):
    """check raw audio files and compute CQT on the fly"""
    print("\n" + "="*60)
    print("TEST B: Raw Audio File Inspection")
    print("="*60)

    try:
        import librosa
    except ImportError:
        print("librosa not installed, skipping audio test")
        return

    import pandas as pd
    metadata = pd.read_csv(index_file)

    for i, (_, row) in enumerate(metadata.iterrows()):
        if i >= num_files:
            break

        audio_path = os.path.join(data_dir, row["audio_file"])
        print(f"\nFile {i+1}: {row['audio_file']}")

        if not os.path.exists(audio_path):
            print(f"  ERROR: File not found!")
            continue

        try:
            # load just 10 seconds
            y, sr = librosa.load(audio_path, sr=22050, duration=10)
            print(f"  Audio loaded: {len(y)} samples, sr={sr}")
            print(f"  Audio range: [{y.min():.4f}, {y.max():.4f}]")
            print(f"  Audio mean: {y.mean():.4f}, std: {y.std():.4f}")

            if np.abs(y).max() < 0.001:
                print("  *** BROKEN: Audio is silent! ***")
            else:
                # compute CQT on the fly
                from conformer_acr.config import SR, HOP_LENGTH, N_CQT_BINS
                cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=252)
                cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
                print(f"  CQT shape: {cqt_db.shape}")
                print(f"  CQT range: [{cqt_db.min():.2f}, {cqt_db.max():.2f}]")
                print(f"  CQT mean: {cqt_db.mean():.2f}, std: {cqt_db.std():.2f}")

                if cqt_db.std() < 0.1:
                    print("  *** WARNING: Very low CQT variance ***")

        except Exception as e:
            print(f"  ERROR: {e}")


def test_dataset_loading(dataset, num_samples=5):
    """check what the dataset actually returns"""
    print("\n" + "="*60)
    print("TEST C: Dataset Loading Check")
    print("="*60)

    print(f"Dataset has {len(dataset)} samples\n")

    all_identical = True
    first_cqt = None

    for i in range(min(num_samples, len(dataset))):
        try:
            cqt, root, bass, qual = dataset[i]
            print(f"Sample {i}:")
            print(f"  CQT shape: {cqt.shape}")
            print(f"  CQT dtype: {cqt.dtype}")
            print(f"  CQT range: [{cqt.min().item():.4f}, {cqt.max().item():.4f}]")
            print(f"  CQT mean: {cqt.mean().item():.4f}, std: {cqt.std().item():.4f}")
            print(f"  Root labels unique: {root.unique().tolist()}")
            print(f"  Quality labels unique: {qual.unique().tolist()}")

            # check if all samples have same CQT
            if first_cqt is None:
                first_cqt = cqt
            elif not torch.allclose(cqt, first_cqt):
                all_identical = False

            if cqt.std().item() < 0.001:
                print("  *** BROKEN: Zero variance CQT! ***")

            print()
        except Exception as e:
            print(f"  ERROR loading sample {i}: {e}\n")

    if all_identical and num_samples > 1:
        print("*** CRITICAL: All samples have IDENTICAL CQT data! ***")


def test_preprocess_function(data_dir, index_file):
    """check the CQT extraction function directly"""
    print("\n" + "="*60)
    print("TEST D: CQT Preprocessing Function")
    print("="*60)

    try:
        from conformer_acr.data.preprocess import extract_cqt
        import pandas as pd

        metadata = pd.read_csv(index_file)
        row = metadata.iloc[0]
        audio_path = os.path.join(data_dir, row["audio_file"])

        print(f"Testing extract_cqt on: {row['audio_file']}")

        if os.path.exists(audio_path):
            cqt = extract_cqt(audio_path)
            print(f"  CQT shape: {cqt.shape}")
            print(f"  CQT dtype: {cqt.dtype}")
            print(f"  CQT range: [{cqt.min().item():.4f}, {cqt.max().item():.4f}]")
            print(f"  CQT mean: {cqt.mean().item():.4f}")
            print(f"  CQT std: {cqt.std().item():.4f}")

            if cqt.std().item() < 0.001:
                print("  *** BROKEN: extract_cqt returns zero-variance data! ***")
            if cqt.min().item() == -80.0 and cqt.max().item() == -80.0:
                print("  *** BROKEN: All values are -80 (silence floor)! ***")
        else:
            print(f"  File not found: {audio_path}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_cqt_content_location(data_dir, index_file, num_files=5):
    """find WHERE in the CQT files the actual content is (not silence)"""
    print("\n" + "="*60)
    print("TEST F: CQT Content Location Analysis")
    print("="*60)
    print("Finding where the non-silent portions are in each file.\n")

    import pandas as pd
    metadata = pd.read_csv(index_file)

    for i, (_, row) in enumerate(metadata.iterrows()):
        if i >= num_files:
            break

        audio_path = os.path.join(data_dir, row["audio_file"])
        cqt_path = audio_path.replace('_mix.flac', '_cqt.pt').replace('.flac', '_cqt.pt')

        if not os.path.exists(cqt_path):
            print(f"File {i+1}: {os.path.basename(cqt_path)} - NOT FOUND")
            continue

        cqt = torch.load(cqt_path, map_location='cpu')
        total_frames = cqt.shape[0]

        # find non-silent frames (where std across freq bins > threshold)
        frame_stds = cqt.std(dim=1)  # std per frame
        non_silent_mask = frame_stds > 0.1
        non_silent_frames = non_silent_mask.sum().item()
        non_silent_pct = 100 * non_silent_frames / total_frames

        # find first and last non-silent frame
        non_silent_indices = torch.where(non_silent_mask)[0]
        if len(non_silent_indices) > 0:
            first_nonsilent = non_silent_indices[0].item()
            last_nonsilent = non_silent_indices[-1].item()
        else:
            first_nonsilent = -1
            last_nonsilent = -1

        # what does center crop get?
        center_start = (total_frames - 512) // 2 if total_frames > 512 else 0
        center_end = center_start + min(512, total_frames)
        center_crop = cqt[center_start:center_end]
        center_std = center_crop.std().item()

        print(f"File {i+1}: {os.path.basename(cqt_path)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Non-silent frames: {non_silent_frames} ({non_silent_pct:.1f}%)")
        print(f"  Non-silent range: frames {first_nonsilent} to {last_nonsilent}")
        print(f"  Center crop (512 frames): {center_start}-{center_end}, std={center_std:.4f}")

        if center_std < 0.1:
            print(f"  *** CENTER CROP IS SILENT! Content is elsewhere in file. ***")
        print()


def test_dataset_vs_raw_file(data_dir, index_file, dataset):
    """compare what Dataset returns vs raw file contents"""
    print("\n" + "="*60)
    print("TEST G: Dataset vs Raw File Comparison")
    print("="*60)

    import pandas as pd
    metadata = pd.read_csv(index_file)
    row = metadata.iloc[0]

    audio_path = os.path.join(data_dir, row["audio_file"])
    cqt_path = audio_path.replace('_mix.flac', '_cqt.pt').replace('.flac', '_cqt.pt')

    print(f"Comparing: {os.path.basename(cqt_path)}\n")

    # load raw file
    if os.path.exists(cqt_path):
        raw_cqt = torch.load(cqt_path, map_location='cpu')
        print(f"Raw file:")
        print(f"  Shape: {raw_cqt.shape}")
        print(f"  Mean: {raw_cqt.mean().item():.4f}, Std: {raw_cqt.std().item():.4f}")
        print(f"  Range: [{raw_cqt.min().item():.4f}, {raw_cqt.max().item():.4f}]")
    else:
        print(f"Raw file not found: {cqt_path}")
        return

    # load through dataset
    dataset_cqt, _, _, _ = dataset[0]
    print(f"\nDataset output:")
    print(f"  Shape: {dataset_cqt.shape}")
    print(f"  Mean: {dataset_cqt.mean().item():.4f}, Std: {dataset_cqt.std().item():.4f}")
    print(f"  Range: [{dataset_cqt.min().item():.4f}, {dataset_cqt.max().item():.4f}]")

    # check if center crop matches
    total = raw_cqt.shape[0]
    center_start = (total - 512) // 2 if total > 512 else 0
    center_end = center_start + min(512, total)
    center_crop = raw_cqt[center_start:center_end]

    print(f"\nManual center crop of raw file:")
    print(f"  Frames {center_start} to {center_end}")
    print(f"  Mean: {center_crop.mean().item():.4f}, Std: {center_crop.std().item():.4f}")

    if torch.allclose(dataset_cqt.double(), center_crop, atol=1e-5):
        print("\n  Dataset output MATCHES manual center crop.")
    else:
        print("\n  Dataset output DIFFERS from manual center crop!")


def test_label_distribution(dataset, num_samples=100):
    """check label distribution in the dataset"""
    print("\n" + "="*60)
    print("TEST E: Label Distribution")
    print("="*60)

    root_counts = {}
    qual_counts = {}
    ignored_frames = 0
    total_frames = 0

    for i in range(min(num_samples, len(dataset))):
        _, root, _, qual = dataset[i]

        for r in root.numpy():
            total_frames += 1
            if r == -100:
                ignored_frames += 1
            else:
                root_counts[r] = root_counts.get(r, 0) + 1

        for q in qual.numpy():
            if q != -100:
                qual_counts[q] = qual_counts.get(q, 0) + 1

    print(f"Analyzed {num_samples} samples, {total_frames} total frames")
    print(f"Ignored frames (label=-100): {ignored_frames} ({100*ignored_frames/total_frames:.1f}%)")

    print(f"\nRoot distribution:")
    for r, c in sorted(root_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  Class {r}: {c} frames")

    print(f"\nQuality distribution:")
    for q, c in sorted(qual_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  Class {q}: {c} frames")

    if ignored_frames / total_frames > 0.9:
        print("\n*** WARNING: >90% of frames are ignored! Model has very little to learn from. ***")


def test_random_inputs(model, num_tests=10):
    """check if model gives varied outputs for random noise"""
    print("\n" + "="*60)
    print("TEST 1: Random Input Response")
    print("="*60)
    print("If model gives same prediction for all random inputs, it's degenerate.\n")

    model.eval()
    quality_preds = []
    root_preds = []

    with torch.no_grad():
        for i in range(num_tests):
            x = torch.randn(1, 100, 252)
            out = model(x)
            q = out["quality"][0, 0, :].argmax().item()
            r = out["root"][0, 0, :].argmax().item()
            quality_preds.append(q)
            root_preds.append(r)
            print(f"  Random input {i}: quality={q}, root={r}")

    q_unique = len(set(quality_preds))
    r_unique = len(set(root_preds))
    print(f"\nUnique quality predictions: {q_unique}/{num_tests}")
    print(f"Unique root predictions: {r_unique}/{num_tests}")

    if q_unique == 1:
        print("WARNING: Model predicts same quality for ALL random inputs!")
    if r_unique == 1:
        print("WARNING: Model predicts same root for ALL random inputs!")

    return q_unique > 1 and r_unique > 1


def test_logit_distribution(model):
    """check if logits are reasonable or degenerate"""
    print("\n" + "="*60)
    print("TEST 2: Logit Distribution Analysis")
    print("="*60)
    print("Checking if output logits have reasonable spread.\n")

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 100, 252)
        out = model(x)

        for head in ["root", "quality", "bass"]:
            logits = out[head][0, 0, :]
            probs = F.softmax(logits, dim=0)

            print(f"{head.upper()}:")
            print(f"  Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
            print(f"  Logit std: {logits.std():.3f}")
            print(f"  Max prob: {probs.max():.3f}")
            print(f"  Entropy: {-(probs * probs.log()).sum():.3f}")

            if probs.max() > 0.99:
                print(f"  WARNING: Model is >99% confident - likely degenerate")
            print()


def test_cqt_statistics(dataset, num_samples=10):
    """compare real CQT stats with random noise"""
    print("\n" + "="*60)
    print("TEST 0: CQT Data Statistics")
    print("="*60)
    print("Comparing real CQT features with random noise.\n")

    all_means = []
    all_stds = []
    all_mins = []
    all_maxs = []

    for i in range(min(num_samples, len(dataset))):
        cqt, _, _, _ = dataset[i]
        all_means.append(cqt.mean().item())
        all_stds.append(cqt.std().item())
        all_mins.append(cqt.min().item())
        all_maxs.append(cqt.max().item())

    print(f"Real CQT data ({num_samples} samples):")
    print(f"  Mean of means: {np.mean(all_means):.4f}")
    print(f"  Mean of stds:  {np.mean(all_stds):.4f}")
    print(f"  Min value:     {np.min(all_mins):.4f}")
    print(f"  Max value:     {np.max(all_maxs):.4f}")

    print(f"\nRandom noise (for comparison):")
    rand = torch.randn(100, 252)
    print(f"  Mean: {rand.mean():.4f}")
    print(f"  Std:  {rand.std():.4f}")
    print(f"  Min:  {rand.min():.4f}")
    print(f"  Max:  {rand.max():.4f}")

    # check if CQT is normalized
    mean_of_means = np.mean(all_means)
    mean_of_stds = np.mean(all_stds)

    if abs(mean_of_means) > 1.0:
        print(f"\nWARNING: CQT mean ({mean_of_means:.2f}) is far from 0!")
        print("  Model trained on unnormalized data behaves differently on normalized random input.")
    if mean_of_stds < 0.5 or mean_of_stds > 2.0:
        print(f"\nWARNING: CQT std ({mean_of_stds:.2f}) differs significantly from 1.0!")


def test_real_vs_random_logits(model, dataset):
    """compare logit distributions between real CQT and random noise"""
    print("\n" + "="*60)
    print("TEST 2B: Real CQT vs Random Noise Logits")
    print("="*60)

    model.eval()
    with torch.no_grad():
        # get a real sample
        cqt_real, _, _, _ = dataset[0]
        cqt_real = cqt_real.unsqueeze(0).float()

        # random noise with same shape
        cqt_rand = torch.randn_like(cqt_real)

        out_real = model(cqt_real)
        out_rand = model(cqt_rand)

        print("\nQuality logits (first frame):")
        print(f"  Real CQT:  {out_real['quality'][0,0,:].numpy().round(2)}")
        print(f"  Random:    {out_rand['quality'][0,0,:].numpy().round(2)}")
        print(f"  Real argmax: {out_real['quality'][0,0,:].argmax().item()}")
        print(f"  Rand argmax: {out_rand['quality'][0,0,:].argmax().item()}")

        print("\nRoot logits (first frame):")
        print(f"  Real CQT:  {out_real['root'][0,0,:].numpy().round(2)}")
        print(f"  Random:    {out_rand['root'][0,0,:].numpy().round(2)}")
        print(f"  Real argmax: {out_real['root'][0,0,:].argmax().item()}")
        print(f"  Rand argmax: {out_rand['root'][0,0,:].argmax().item()}")

        # check max logit magnitude
        real_max = out_real['quality'][0,0,:].max().item()
        rand_max = out_rand['quality'][0,0,:].max().item()
        print(f"\nMax quality logit - Real: {real_max:.2f}, Random: {rand_max:.2f}")

        if real_max > rand_max + 5:
            print("WARNING: Model is much more confident on real data!")
            print("  This suggests it learned a spurious pattern in CQT features.")


def test_real_data_predictions(model, dataset, vocab_mapper, num_samples=5):
    """check predictions on actual data samples"""
    print("\n" + "="*60)
    print("TEST 3: Real Data Predictions")
    print("="*60)
    print("Checking if model gives varied predictions on real data.\n")

    qualities = vocab_mapper.qualities
    roots = list(vocab_mapper.root_to_idx.keys())

    model.eval()
    all_q_preds = []
    all_r_preds = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            cqt, root_labels, bass_labels, qual_labels = dataset[i]
            cqt = cqt.unsqueeze(0).float()  # add batch dim, ensure float32

            out = model(cqt)

            # get predictions for middle frame
            mid = cqt.shape[1] // 2
            q_pred = out["quality"][0, mid, :].argmax().item()
            r_pred = out["root"][0, mid, :].argmax().item()
            q_true = qual_labels[mid].item()
            r_true = root_labels[mid].item()

            all_q_preds.append(q_pred)
            all_r_preds.append(r_pred)

            q_pred_name = qualities[q_pred] if q_pred < len(qualities) else "?"
            r_pred_name = roots[r_pred] if r_pred < len(roots) else "?"
            q_true_name = qualities[q_true] if 0 <= q_true < len(qualities) else "IGN"
            r_true_name = roots[r_true] if 0 <= r_true < len(roots) else "IGN"

            print(f"  Sample {i}:")
            print(f"    Quality: pred={q_pred_name:8s} true={q_true_name}")
            print(f"    Root:    pred={r_pred_name:8s} true={r_true_name}")

    print(f"\nPrediction diversity on {num_samples} samples:")
    print(f"  Unique quality preds: {len(set(all_q_preds))}")
    print(f"  Unique root preds: {len(set(all_r_preds))}")

    from collections import Counter
    print(f"\nQuality prediction counts: {dict(Counter(all_q_preds))}")
    print(f"Root prediction counts: {dict(Counter(all_r_preds))}")


def test_gradient_flow(model):
    """check if gradients flow through all heads"""
    print("\n" + "="*60)
    print("TEST 4: Gradient Flow")
    print("="*60)
    print("Checking if gradients flow to all output heads.\n")

    model.train()
    x = torch.randn(1, 50, 252, requires_grad=True)
    out = model(x)

    for head in ["root", "quality", "bass"]:
        # compute loss for this head
        logits = out[head]
        target = torch.randint(0, logits.shape[-1], (1, 50))
        loss = F.cross_entropy(logits.transpose(1, 2), target)

        model.zero_grad()
        loss.backward(retain_graph=True)

        # check gradients on output head
        head_layer = getattr(model, f"head_{head}")
        grad_norm = head_layer.weight.grad.norm().item()
        print(f"  {head}: gradient norm = {grad_norm:.6f}")

        if grad_norm < 1e-7:
            print(f"    WARNING: Near-zero gradient for {head} head!")

    model.eval()


def test_weight_statistics(model):
    """check if weights are reasonable"""
    print("\n" + "="*60)
    print("TEST 5: Weight Statistics")
    print("="*60)

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  {name}: mean={param.mean():.4f}, std={param.std():.4f}, "
                  f"min={param.min():.4f}, max={param.max():.4f}")


def test_prediction_distribution(model, dataset, num_samples=100):
    """get full prediction distribution over many samples"""
    print("\n" + "="*60)
    print("TEST 6: Prediction Distribution (100 samples)")
    print("="*60)

    model.eval()
    q_preds = []
    r_preds = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            cqt, root_labels, bass_labels, qual_labels = dataset[i]
            cqt = cqt.unsqueeze(0).float()  # ensure float32
            out = model(cqt)

            # get all frame predictions
            q = out["quality"][0].argmax(dim=-1).numpy()
            r = out["root"][0].argmax(dim=-1).numpy()
            q_preds.extend(q.tolist())
            r_preds.extend(r.tolist())

    from collections import Counter
    q_counts = Counter(q_preds)
    r_counts = Counter(r_preds)

    total = len(q_preds)
    print(f"\nTotal frames analyzed: {total}\n")

    print("Quality prediction distribution:")
    for cls, count in sorted(q_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  Class {cls:2d}: {count:6d} ({pct:5.1f}%)")

    print("\nRoot prediction distribution:")
    for cls, count in sorted(r_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  Class {cls:2d}: {count:6d} ({pct:5.1f}%)")

    # mode collapse check
    q_max_pct = 100 * max(q_counts.values()) / total
    r_max_pct = 100 * max(r_counts.values()) / total

    print(f"\n*** MODE COLLAPSE CHECK ***")
    print(f"Most predicted quality: {q_max_pct:.1f}%")
    print(f"Most predicted root: {r_max_pct:.1f}%")

    if q_max_pct > 80:
        print("FAIL: Quality predictions collapsed to one class!")
    else:
        print("PASS: Quality predictions are distributed")

    if r_max_pct > 80:
        print("FAIL: Root predictions collapsed to one class!")
    else:
        print("PASS: Root predictions are distributed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--index-file", type=str, required=True)
    args = parser.parse_args()

    print("="*60)
    print("CONFORMER-ACR DIAGNOSTIC REPORT")
    print("="*60)

    # load vocab
    cache_path = os.path.join(args.data_dir, "vocab_cache.json")
    with open(cache_path, 'r') as f:
        cache = json.load(f)

    # import here to avoid issues
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train import VocabMapper

    vocab_mapper = VocabMapper(cache["qualities"])

    # load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = ConformerACR(
        num_roots=vocab_mapper.num_roots,
        num_qualities=vocab_mapper.num_qualities,
        num_bass=vocab_mapper.num_bass,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded epoch {ckpt['epoch']}")

    # load dataset
    dataset = AAMDataset(
        index_file=args.index_file,
        audio_dir=args.data_dir,
        vocab_mapper=vocab_mapper,
        max_seq_len=512,
        random_crop=False,
    )
    print(f"Dataset: {len(dataset)} samples")

    # run data tests FIRST - if data is broken, nothing else matters
    print("\n" + "#"*60)
    print("# PART 1: DATA PIPELINE TESTS")
    print("#"*60)
    test_raw_cqt_files(args.data_dir, num_files=5)
    test_audio_files(args.data_dir, args.index_file, num_files=3)
    test_preprocess_function(args.data_dir, args.index_file)
    test_dataset_loading(dataset, num_samples=5)
    test_label_distribution(dataset, num_samples=100)
    test_cqt_content_location(args.data_dir, args.index_file, num_files=5)
    test_dataset_vs_raw_file(args.data_dir, args.index_file, dataset)
    test_cqt_statistics(dataset, num_samples=10)

    print("\n" + "#"*60)
    print("# PART 2: MODEL TESTS")
    print("#"*60)
    test_random_inputs(model)
    test_logit_distribution(model)
    test_gradient_flow(model)
    test_weight_statistics(model)
    test_real_vs_random_logits(model, dataset)

    print("\n" + "#"*60)
    print("# PART 3: PREDICTION TESTS")
    print("#"*60)
    test_real_data_predictions(model, dataset, vocab_mapper, num_samples=10)
    test_prediction_distribution(model, dataset, num_samples=100)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
