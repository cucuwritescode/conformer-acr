"""
conformer_acr.training.losses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom loss functions for chord recognition training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    Implements the focal loss from *Lin et al., 2017*:

    .. math::
        FL(p_t) = -\\alpha_t \\, (1 - p_t)^\\gamma \\, \\log(p_t)

    This down-weights easy (well-classified) examples and focuses
    training on hard negatives — essential for chord recognition
    where ``N`` (no-chord) dominates.

    Parameters
    ----------
    weight : Tensor, optional
        Per-class weights of shape ``(num_classes,)``. Use inverse frequency.
    gamma : float
        Focusing parameter — higher values increase the effect
        (default: ``2.0``).
    reduction : str
        ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        inputs : Tensor, shape ``(N, C)`` or ``(N, C, T)``
            Raw logits (pre-softmax).
        targets : Tensor, shape ``(N,)`` or ``(N, T)``
            Ground-truth class indices.

        Returns
        -------
        Tensor
            Scalar loss (or per-sample if ``reduction='none'``).
        """
        #BUG FIX: old code computed pt = exp(-weighted_ce) = p^weight, not p
        #this broke focal modulation when combined with class weights

        #get pt from unweighted log-softmax (the actual probability)
        if inputs.dim() == 3:
            #(N, C, T) case: log_softmax along class dim
            log_probs = F.log_softmax(inputs, dim=1)
            log_pt = log_probs.gather(1, targets.unsqueeze(1).clamp(min=0)).squeeze(1)
        else:
            #(N, C) standard case
            log_probs = F.log_softmax(inputs, dim=1)
            log_pt = log_probs.gather(1, targets.unsqueeze(1).clamp(min=0)).squeeze(1)

        pt = log_pt.exp()
        ce_loss = -log_pt  #unweighted cross-entropy

        #focal modulation: down-weight easy examples
        focal_weight = (1.0 - pt) ** self.gamma

        #apply class weights separately (not baked into pt)
        if self.weight is not None:
            weight_targets = targets.clamp(min=0, max=self.weight.size(0) - 1)
            alpha = self.weight[weight_targets]
            focal_loss = alpha * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        #mask out ignored indices (padding and N class use -100)
        valid = (targets >= 0)
        focal_loss = focal_loss * valid.float()

        if self.reduction == "mean":
            n_valid = valid.sum()
            if n_valid == 0:
                return focal_loss.new_zeros((), requires_grad=True)
            return focal_loss.sum() / n_valid
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
