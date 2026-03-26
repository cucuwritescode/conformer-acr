"""
tests/test_training_integration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration test: run actual training loop with mock data.
This catches issues the unit tests might miss.

Run with: pytest tests/test_training_integration.py -v -s
"""

import torch
import torch.nn as nn
import numpy as np
import pytest

from conformer_acr.training.losses import FocalLoss
from conformer_acr.models.conformer import ConformerACR


class MockDataset:
    """fake dataset with known class distribution"""

    def __init__(self, num_samples=100, seq_len=50, num_qualities=12):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_qualities = num_qualities
        self.num_roots = 13
        self.num_bass = 13

        #create imbalanced distribution (like real chord data)
        #adjust probs to match num_qualities
        probs = np.array([0.30, 0.25, 0.15, 0.10, 0.10, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.001])
        self.quality_probs = probs[:num_qualities]
        self.quality_probs = self.quality_probs / self.quality_probs.sum()  #normalize

    def __len__(self):
        return self.num_samples

    def get_batch(self, batch_size=8):
        #random CQT features
        cqt = torch.randn(batch_size, self.seq_len, 252)

        #imbalanced labels (some classes rare)
        qual = torch.from_numpy(
            np.random.choice(self.num_qualities, size=(batch_size, self.seq_len),
                           p=self.quality_probs)
        ).long()
        root = torch.randint(0, self.num_roots, (batch_size, self.seq_len))
        bass = torch.randint(0, self.num_bass, (batch_size, self.seq_len))

        #add some padding (-100) like real data
        for i in range(batch_size):
            pad_start = np.random.randint(self.seq_len - 10, self.seq_len)
            qual[i, pad_start:] = -100
            root[i, pad_start:] = -100
            bass[i, pad_start:] = -100

        lengths = torch.tensor([self.seq_len] * batch_size)

        return {
            "cqt": cqt,
            "qual": qual,
            "root": root,
            "bass": bass,
            "lengths": lengths,
        }


def compute_test_class_weights(num_classes, class_counts):
    """replicate the weight computation from train.py"""
    eps = 1.0
    weights = class_counts.sum() / (len(class_counts) * (class_counts + eps))

    #exclude zero-count classes from normalization
    mask = class_counts > 0
    if mask.sum() > 0:
        weights[mask] = weights[mask] / weights[mask].mean()
    weights[~mask] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


class TestTrainingIntegration:
    """full training loop tests"""

    def test_model_learns_from_imbalanced_data(self):
        """THE CRITICAL TEST: model should learn all classes, not collapse"""
        torch.manual_seed(42)
        np.random.seed(42)

        device = torch.device("cpu")
        dataset = MockDataset(num_samples=50, seq_len=30, num_qualities=6)

        #create model
        model = ConformerACR(
            d_model=64,  #small for fast test
            n_heads=2,
            n_layers=2,
            num_roots=13,
            num_qualities=6,
            num_bass=13,
        ).to(device)

        #simulate class counts (imbalanced)
        qual_counts = np.array([3000.0, 2500.0, 1500.0, 500.0, 300.0, 200.0])
        qual_weights = compute_test_class_weights(6, qual_counts)

        #create loss functions
        loss_fns = {
            "root": FocalLoss(gamma=2.0, reduction="mean"),
            "quality": FocalLoss(weight=qual_weights, gamma=2.0, reduction="mean"),
            "bass": FocalLoss(gamma=2.0, reduction="mean"),
        }

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        #track predictions per class over training
        initial_preds = {i: 0 for i in range(6)}
        final_preds = {i: 0 for i in range(6)}

        #training loop
        model.train()
        losses = []

        for epoch in range(20):  #20 mini-epochs
            batch = dataset.get_batch(batch_size=8)

            cqt = batch["cqt"].to(device)
            qual = batch["qual"].to(device)
            root = batch["root"].to(device)
            bass = batch["bass"].to(device)

            max_len = cqt.size(1)
            mask = torch.zeros(cqt.size(0), max_len, dtype=torch.bool, device=device)

            optimizer.zero_grad()

            out = model(cqt, mask=mask)

            loss_root = loss_fns["root"](out["root"].transpose(1, 2), root)
            loss_qual = loss_fns["quality"](out["quality"].transpose(1, 2), qual)
            loss_bass = loss_fns["bass"](out["bass"].transpose(1, 2), bass)
            loss = loss_root + loss_qual + loss_bass

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            #track prediction distribution
            preds = out["quality"].argmax(dim=-1)
            valid = (qual >= 0)

            if epoch == 0:
                for p in preds[valid].cpu().numpy():
                    initial_preds[p] += 1
            if epoch == 19:
                for p in preds[valid].cpu().numpy():
                    final_preds[p] += 1

        #CRITICAL CHECKS

        #1. loss should decrease
        assert losses[-1] < losses[0], \
            f"loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

        #2. loss should not be NaN
        assert not any(np.isnan(losses)), "loss should never be NaN"

        #3. model should NOT collapse to predicting one class
        #check that at least 3 different classes are predicted
        classes_predicted = sum(1 for c in final_preds.values() if c > 0)
        assert classes_predicted >= 3, \
            f"MODE COLLAPSE DETECTED: only {classes_predicted} classes predicted. " \
            f"Distribution: {final_preds}"

        print(f"\nloss: {losses[0]:.4f} -> {losses[-1]:.4f}")
        print(f"initial pred distribution: {initial_preds}")
        print(f"final pred distribution: {final_preds}")
        print(f"classes with predictions: {classes_predicted}/6")

    def test_rare_class_gets_gradients(self):
        """rare classes should receive meaningful gradients"""
        torch.manual_seed(42)

        model = ConformerACR(
            d_model=64, n_heads=2, n_layers=2,
            num_roots=13, num_qualities=6, num_bass=13
        )

        #extreme imbalance: class 5 is very rare
        qual_counts = np.array([10000.0, 8000.0, 5000.0, 1000.0, 500.0, 50.0])
        qual_weights = compute_test_class_weights(6, qual_counts)

        print(f"\nclass weights: {qual_weights}")

        loss_fn = FocalLoss(weight=qual_weights, gamma=2.0, reduction="mean")

        #batch with ONE sample from rare class 5
        cqt = torch.randn(4, 20, 252)
        targets = torch.tensor([
            [0]*20,  #all class 0 (common)
            [1]*20,  #all class 1 (common)
            [0]*20,  #all class 0 (common)
            [5]*20,  #all class 5 (RARE)
        ])

        out = model(cqt, mask=None)
        logits = out["quality"].transpose(1, 2)

        #compute loss
        loss = loss_fn(logits, targets)
        loss.backward()

        #check gradients exist for quality head
        grad = model.head_quality.weight.grad
        assert grad is not None, "gradients should exist"
        assert grad.abs().sum() > 0, "gradients should be non-zero"

        #the rare class (5) should have meaningful gradient
        rare_class_grad = grad[5].abs().mean()
        common_class_grad = grad[0].abs().mean()

        print(f"rare class 5 grad magnitude: {rare_class_grad:.6f}")
        print(f"common class 0 grad magnitude: {common_class_grad:.6f}")

        #rare class should have gradient (not zero)
        assert rare_class_grad > 1e-6, \
            f"rare class gradient too small: {rare_class_grad}"

    def test_no_nan_with_extreme_weights(self):
        """no NaN even with very extreme class weights"""
        torch.manual_seed(42)

        #extreme weights like you might see with 1000:1 imbalance
        weights = torch.tensor([0.01, 0.02, 0.5, 1.0, 5.0, 50.0])

        loss_fn = FocalLoss(weight=weights, gamma=2.0, reduction="mean")

        #many batches to stress test
        for _ in range(100):
            logits = torch.randn(8, 6, 30)
            targets = torch.randint(0, 6, (8, 30))
            targets[:, -5:] = -100  #some padding

            loss = loss_fn(logits, targets)

            assert not torch.isnan(loss), "loss should never be NaN"
            assert not torch.isinf(loss), "loss should never be inf"
            assert loss >= 0, "loss should be non-negative"

    def test_trainer_integration(self):
        """test with actual Trainer class"""
        from conformer_acr.training.trainer import Trainer

        torch.manual_seed(42)
        device = torch.device("cpu")

        model = ConformerACR(
            d_model=64, n_heads=2, n_layers=2,
            num_roots=13, num_qualities=6, num_bass=13
        ).to(device)

        qual_counts = np.array([3000.0, 2500.0, 1500.0, 500.0, 300.0, 200.0])
        qual_weights = compute_test_class_weights(6, qual_counts).to(device)

        loss_fns = {
            "root": FocalLoss(gamma=2.0, reduction="mean"),
            "quality": FocalLoss(weight=qual_weights, gamma=2.0, reduction="mean"),
            "bass": FocalLoss(gamma=2.0, reduction="mean"),
        }

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fns=loss_fns,
            device=device,
            checkpoint_dir=None,
            rank=0,
            use_ddp=False,
        )

        #create a simple mock dataloader
        class MockLoader:
            def __init__(self):
                self.dataset = MockDataset(num_samples=20, seq_len=30, num_qualities=6)

            def __iter__(self):
                for _ in range(5):
                    yield self.dataset.get_batch(batch_size=4)

        loader = MockLoader()

        #run one training epoch
        train_loss = trainer._train_epoch(loader)

        assert not np.isnan(train_loss), "training loss should not be NaN"
        assert train_loss > 0, "training loss should be positive"
        print(f"\ntrainer integration test - epoch loss: {train_loss:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
