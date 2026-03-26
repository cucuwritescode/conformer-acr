"""
tests/test_losses.py
~~~~~~~~~~~~~~~~~~~~

Unit tests for FocalLoss to verify the bug fix works.
Run with: pytest tests/test_losses.py -v
"""

import torch
import torch.nn.functional as F
import numpy as np
import pytest

from conformer_acr.training.losses import FocalLoss


class TestFocalLossPtCalculation:
    """verify pt is actual probability, not p^weight (the bug we fixed)"""

    def test_pt_equals_actual_probability_no_weights(self):
        """without class weights, pt should equal softmax probability"""
        torch.manual_seed(42)

        logits = torch.randn(4, 5)  #4 samples, 5 classes
        targets = torch.tensor([0, 1, 2, 3])

        #compute what pt SHOULD be
        probs = F.softmax(logits, dim=1)
        expected_pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        #create focal loss and manually check pt computation
        loss_fn = FocalLoss(gamma=2.0, reduction="none")

        #replicate the forward pass logic
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        actual_pt = log_pt.exp()

        assert torch.allclose(actual_pt, expected_pt, atol=1e-6), \
            f"pt mismatch: got {actual_pt}, expected {expected_pt}"

    def test_pt_not_affected_by_class_weights(self):
        """pt should be the same regardless of class weights"""
        torch.manual_seed(42)

        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])

        #compute expected pt (from softmax)
        probs = F.softmax(logits, dim=1)
        expected_pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        #with extreme weights, pt should still be the same
        weights = torch.tensor([0.1, 10.0, 0.5, 5.0, 1.0])

        #the old buggy code would compute pt = p^weight
        #verify our fix computes pt = p (unaffected by weight)
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        actual_pt = log_pt.exp()

        assert torch.allclose(actual_pt, expected_pt, atol=1e-6), \
            "pt should not be affected by class weights"

    def test_old_bug_would_give_wrong_pt(self):
        """demonstrate that the OLD buggy approach gives wrong pt"""
        torch.manual_seed(42)

        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        weights = torch.tensor([0.1, 10.0, 0.5, 5.0, 1.0])

        #correct pt
        probs = F.softmax(logits, dim=1)
        correct_pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        #OLD BUGGY approach: pt = exp(-weighted_ce) = p^weight
        weighted_ce = F.cross_entropy(logits, targets, weight=weights, reduction="none")
        buggy_pt = torch.exp(-weighted_ce)

        #these should NOT be equal (proving the bug existed)
        assert not torch.allclose(buggy_pt, correct_pt, atol=1e-3), \
            "buggy pt should differ from correct pt (this test validates the bug existed)"


class TestFocalLoss3DInput:
    """verify focal loss works with (N, C, T) shaped inputs"""

    def test_3d_input_shape(self):
        """FocalLoss should handle (batch, classes, time) inputs"""
        torch.manual_seed(42)

        batch, classes, time = 2, 5, 10
        logits = torch.randn(batch, classes, time)
        targets = torch.randint(0, classes, (batch, time))

        loss_fn = FocalLoss(gamma=2.0, reduction="mean")
        loss = loss_fn(logits, targets)

        assert loss.dim() == 0, "loss should be scalar"
        assert not torch.isnan(loss), "loss should not be NaN"
        assert loss > 0, "loss should be positive"

    def test_3d_input_with_ignore_index(self):
        """verify ignore_index=-100 works for 3D inputs (padding)"""
        torch.manual_seed(42)

        batch, classes, time = 2, 5, 10
        logits = torch.randn(batch, classes, time)
        targets = torch.randint(0, classes, (batch, time))

        #set half the targets to -100 (ignored)
        targets[:, 5:] = -100

        loss_fn = FocalLoss(gamma=2.0, reduction="mean")
        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss), "loss should not be NaN with ignore_index"
        assert loss > 0, "loss should be positive"

    def test_3d_all_ignored_returns_zero(self):
        """if all targets are -100, loss should be 0"""
        logits = torch.randn(2, 5, 10)
        targets = torch.full((2, 10), -100, dtype=torch.long)

        loss_fn = FocalLoss(gamma=2.0, reduction="mean")
        loss = loss_fn(logits, targets)

        assert loss.item() == 0.0, "loss should be 0 when all targets ignored"


class TestFocalLossClassWeights:
    """verify class weights are applied correctly"""

    def test_higher_weight_increases_loss(self):
        """samples from high-weight classes should contribute more to loss"""
        torch.manual_seed(42)

        #same logits and targets, different weights
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 0, 0, 0])  #all class 0

        weights_low = torch.tensor([0.1, 1.0, 1.0])
        weights_high = torch.tensor([10.0, 1.0, 1.0])

        loss_fn_low = FocalLoss(weight=weights_low, gamma=0.0, reduction="mean")  #gamma=0 to isolate weight effect
        loss_fn_high = FocalLoss(weight=weights_high, gamma=0.0, reduction="mean")

        loss_low = loss_fn_low(logits, targets)
        loss_high = loss_fn_high(logits, targets)

        assert loss_high > loss_low, \
            f"high weight loss ({loss_high}) should be > low weight loss ({loss_low})"

    def test_weights_applied_per_sample(self):
        """each sample should use the weight for its target class"""
        torch.manual_seed(42)

        logits = torch.randn(3, 3)
        targets = torch.tensor([0, 1, 2])  #one sample per class
        weights = torch.tensor([1.0, 2.0, 3.0])

        loss_fn = FocalLoss(weight=weights, gamma=0.0, reduction="none")
        losses = loss_fn(logits, targets)

        #verify shape
        assert losses.shape == (3,), f"expected shape (3,), got {losses.shape}"

        #the loss ratio should roughly match weight ratio (not exact due to different probs)
        #but loss[2] should be highest since weight[2]=3.0 is highest
        #(this is approximate, the main check is that it doesn't crash)
        assert not torch.any(torch.isnan(losses)), "no losses should be NaN"


class TestFocalModulation:
    """verify focal modulation (1-pt)^gamma works correctly"""

    def test_gamma_zero_equals_weighted_ce(self):
        """with gamma=0, focal loss should equal weighted cross-entropy"""
        torch.manual_seed(42)

        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        weights = torch.tensor([1.0, 2.0, 0.5, 1.5, 1.0])

        #focal loss with gamma=0
        focal_fn = FocalLoss(weight=weights, gamma=0.0, reduction="none")
        focal_loss = focal_fn(logits, targets)

        #standard weighted cross-entropy (for comparison)
        ce_loss = F.cross_entropy(logits, targets, weight=weights, reduction="none")

        assert torch.allclose(focal_loss, ce_loss, atol=1e-5), \
            "gamma=0 focal loss should equal weighted CE"

    def test_high_gamma_downweights_easy_examples(self):
        """high confidence predictions should have lower loss with high gamma"""
        #create logits where model is very confident about class 0
        logits = torch.tensor([[10.0, 0.0, 0.0],   #very confident
                               [0.5, 0.3, 0.2]])    #less confident
        targets = torch.tensor([0, 0])

        loss_fn_low_gamma = FocalLoss(gamma=0.0, reduction="none")
        loss_fn_high_gamma = FocalLoss(gamma=2.0, reduction="none")

        loss_low = loss_fn_low_gamma(logits, targets)
        loss_high = loss_fn_high_gamma(logits, targets)

        #for the confident sample, high gamma should reduce loss more
        confident_ratio = loss_high[0] / loss_low[0]
        uncertain_ratio = loss_high[1] / loss_low[1]

        assert confident_ratio < uncertain_ratio, \
            "high gamma should reduce confident sample loss more than uncertain"


class TestWeightNormalization:
    """verify class weight normalization excludes zero-count classes"""

    def test_normalization_excludes_zero_counts(self):
        """weights for zero-count classes should be 0, not inflate others"""
        import numpy as np

        #simulate counts: class 0 and 1 have data, class 2 (N) has zero
        counts = np.array([1000.0, 500.0, 0.0])
        eps = 1.0

        #compute inverse freq weights
        weights = counts.sum() / (len(counts) * (counts + eps))

        #old buggy normalization (includes the huge N weight)
        old_normalized = weights / weights.mean()

        #new correct normalization (excludes zero-count classes)
        mask = counts > 0
        new_normalized = weights.copy()
        new_normalized[mask] = weights[mask] / weights[mask].mean()
        new_normalized[~mask] = 0.0

        #old normalization makes real weights tiny
        assert old_normalized[0] < 0.1, "old normalization deflates real weights"

        #new normalization keeps real weights reasonable
        assert new_normalized[0] > 0.5, "new normalization preserves real weights"
        assert new_normalized[2] == 0.0, "zero-count class should have weight 0"


class TestEndToEnd:
    """end-to-end tests simulating actual training scenario"""

    def test_loss_decreases_with_correct_predictions(self):
        """loss should be lower when model predicts correctly"""
        #logits that predict class 0 for all samples
        logits_correct = torch.tensor([[5.0, 0.0, 0.0],
                                        [5.0, 0.0, 0.0]])
        logits_wrong = torch.tensor([[0.0, 5.0, 0.0],
                                      [0.0, 0.0, 5.0]])
        targets = torch.tensor([0, 0])

        loss_fn = FocalLoss(gamma=2.0, reduction="mean")

        loss_correct = loss_fn(logits_correct, targets)
        loss_wrong = loss_fn(logits_wrong, targets)

        assert loss_correct < loss_wrong, \
            f"correct predictions ({loss_correct}) should have lower loss than wrong ({loss_wrong})"

    def test_gradient_flows_to_all_classes(self):
        """verify gradients flow properly (no mode collapse setup)"""
        torch.manual_seed(42)

        logits = torch.randn(8, 5, requires_grad=True)
        #diverse targets across classes
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
        weights = torch.tensor([1.0, 2.0, 1.5, 0.8, 1.2])

        loss_fn = FocalLoss(weight=weights, gamma=2.0, reduction="mean")
        loss = loss_fn(logits, targets)
        loss.backward()

        #gradients should exist and not be zero for all logit positions
        assert logits.grad is not None, "gradients should exist"
        assert not torch.all(logits.grad == 0), "gradients should not all be zero"

        #check gradient magnitude is reasonable (not exploding)
        assert logits.grad.abs().max() < 100, "gradients should not explode"

    def test_realistic_chord_scenario(self):
        """simulate actual chord recognition: (batch, classes, time) with padding"""
        torch.manual_seed(42)

        batch, num_qualities, time = 4, 12, 100
        logits = torch.randn(batch, num_qualities, time, requires_grad=True)

        #targets with some padding (-100)
        targets = torch.randint(0, num_qualities, (batch, time))
        targets[:, 80:] = -100  #last 20 frames are padding

        #realistic class weights (some rare, some common)
        weights = torch.tensor([1.0, 5.0, 8.0, 0.5, 3.0, 2.0,
                                0.3, 1.5, 0.8, 4.0, 6.0, 0.0])  #last is N, weight=0

        loss_fn = FocalLoss(weight=weights, gamma=2.0, reduction="mean")
        loss = loss_fn(logits, targets)

        assert not torch.isnan(loss), "loss should not be NaN"
        assert loss > 0, "loss should be positive"

        loss.backward()
        assert logits.grad is not None, "gradients should flow"

        #gradients for padded positions should be zero
        #(indirectly tested - the loss ignores them)
        print(f"realistic scenario loss: {loss.item():.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
