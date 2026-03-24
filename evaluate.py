#!/usr/bin/env python3
"""
evaluate.py - Class-wise evaluation for ConformerACR

Generates a performance report with:
- Global accuracy for root, quality, bass
- Class-wise recall for quality (the long-tail check)
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  #fallback: no progress bar

from conformer_acr.models.conformer import ConformerACR
from conformer_acr.data.dataset import AAMDataset, pad_collate_fn
from train import VocabMapper, scan_dataset_for_vocab


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #load vocab from cache
    cache_path = os.path.join(args.data_dir, "vocab_cache.json")
    if os.path.exists(cache_path):
        print(f"Loading vocabulary from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        vocab_mapper = VocabMapper(cache["qualities"])
    else:
        raise FileNotFoundError(f"Vocab cache not found: {cache_path}")

    #load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = ConformerACR(
        num_roots=vocab_mapper.num_roots,
        num_qualities=vocab_mapper.num_qualities,
        num_bass=vocab_mapper.num_bass,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    #load dataset
    dataset = AAMDataset(
        index_file=args.index_file,
        audio_dir=args.data_dir,
        vocab_mapper=vocab_mapper,
        max_seq_len=args.max_seq_len,
        random_crop=False,  #deterministic for eval
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=args.num_workers,
    )
    print(f"Evaluating on {len(dataset)} samples...")

    #metrics storage
    results = {
        'root': {'correct': 0, 'total': 0},
        'quality': {'correct': 0, 'total': 0},
        'bass': {'correct': 0, 'total': 0},
        'full_chord': {'correct': 0, 'total': 0},  #all three heads correct
    }

    #class-wise recall for quality (the long-tail check)
    qual_names = vocab_mapper.qualities
    qual_stats = {q: {'correct': 0, 'total': 0} for q in qual_names}

    #class-wise recall for root
    root_names = list(vocab_mapper.root_to_idx.keys())
    root_stats = {r: {'correct': 0, 'total': 0} for r in root_names}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            cqt = batch["cqt"].float().to(device)
            lengths = batch["lengths"].to(device)
            target_root = batch["root"].to(device)
            target_qual = batch["qual"].to(device)
            target_bass = batch["bass"].to(device)

            #create padding mask
            max_len = cqt.size(1)
            mask = (torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)).bool()

            #inference
            out = model(cqt, mask=mask)
            pred_root = out["root"].argmax(dim=-1)
            pred_qual = out["quality"].argmax(dim=-1)
            pred_bass = out["bass"].argmax(dim=-1)

            #valid mask (ignore padding, value=-100)
            valid = (target_root != -100)

            #global metrics
            results['root']['correct'] += (pred_root[valid] == target_root[valid]).sum().item()
            results['root']['total'] += valid.sum().item()
            results['quality']['correct'] += (pred_qual[valid] == target_qual[valid]).sum().item()
            results['quality']['total'] += valid.sum().item()
            results['bass']['correct'] += (pred_bass[valid] == target_bass[valid]).sum().item()
            results['bass']['total'] += valid.sum().item()

            #full chord accuracy (all three heads correct)
            full_match = (pred_root == target_root) & (pred_qual == target_qual) & (pred_bass == target_bass) & valid
            results['full_chord']['correct'] += full_match.sum().item()
            results['full_chord']['total'] += valid.sum().item()

            #class-wise quality recall
            for i, q_name in enumerate(qual_names):
                q_mask = (target_qual == i) & valid
                qual_stats[q_name]['correct'] += (pred_qual[q_mask] == i).sum().item()
                qual_stats[q_name]['total'] += q_mask.sum().item()

            #class-wise root recall
            for i, r_name in enumerate(root_names):
                r_mask = (target_root == i) & valid
                root_stats[r_name]['correct'] += (pred_root[r_mask] == i).sum().item()
                root_stats[r_name]['total'] += r_mask.sum().item()

    #compute accuracies
    accuracies = {}
    for head in ['root', 'quality', 'bass', 'full_chord']:
        accuracies[head] = results[head]['correct'] / max(results[head]['total'], 1)

    #print report
    print("\n" + "=" * 50)
    print("GLOBAL PERFORMANCE")
    print("=" * 50)
    for head in ['root', 'quality', 'bass']:
        print(f"{head.capitalize():10} Accuracy: {accuracies[head]:.4f} ({results[head]['correct']}/{results[head]['total']})")
    print("-" * 50)
    print(f"{'Full Chord':10} Accuracy: {accuracies['full_chord']:.4f} ({results['full_chord']['correct']}/{results['full_chord']['total']})")

    print("\n" + "=" * 50)
    print("QUALITY CLASS-WISE RECALL (THE LONG TAIL)")
    print("=" * 50)
    print(f"{'Quality':<12} | {'Recall':<10} | {'Correct':<10} | {'Total':<10}")
    print("-" * 50)
    for q_name in qual_names:
        total = qual_stats[q_name]['total']
        correct = qual_stats[q_name]['correct']
        if total > 0:
            recall = correct / total
            print(f"{q_name:<12} | {recall:.4f}     | {correct:<10} | {total:<10}")
        else:
            print(f"{q_name:<12} | N/A        | 0          | 0")

    print("\n" + "=" * 50)
    print("ROOT CLASS-WISE RECALL")
    print("=" * 50)
    print(f"{'Root':<12} | {'Recall':<10} | {'Correct':<10} | {'Total':<10}")
    print("-" * 50)
    for r_name in root_names:
        total = root_stats[r_name]['total']
        correct = root_stats[r_name]['correct']
        if total > 0:
            recall = correct / total
            print(f"{r_name:<12} | {recall:.4f}     | {correct:<10} | {total:<10}")
        else:
            print(f"{r_name:<12} | N/A        | 0          | 0")
    print("=" * 50)

    #save results to JSON
    output = {
        'checkpoint': args.checkpoint,
        'index_file': args.index_file,
        'num_samples': len(dataset),
        'accuracies': accuracies,
        'quality_recall': {q: qual_stats[q]['correct'] / max(qual_stats[q]['total'], 1)
                          for q in qual_names if qual_stats[q]['total'] > 0},
        'root_recall': {r: root_stats[r]['correct'] / max(root_stats[r]['total'], 1)
                       for r in root_names if root_stats[r]['total'] > 0},
    }

    out_path = args.output if args.output else "eval_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ConformerACR model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--index-file", type=str, required=True,
                        help="CSV file with audio_file and label_file columns")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory containing audio and label files")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results (default: eval_results.json)")
    args = parser.parse_args()
    evaluate(args)
