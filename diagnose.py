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
import torch
import torch.nn.functional as F
import numpy as np

from conformer_acr.models.conformer import ConformerACR
from conformer_acr.data.dataset import AAMDataset, pad_collate_fn
from torch.utils.data import DataLoader


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

    # run tests
    test_cqt_statistics(dataset, num_samples=10)
    test_random_inputs(model)
    test_logit_distribution(model)
    test_gradient_flow(model)
    test_weight_statistics(model)
    test_real_vs_random_logits(model, dataset)
    test_real_data_predictions(model, dataset, vocab_mapper, num_samples=10)
    test_prediction_distribution(model, dataset, num_samples=100)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
