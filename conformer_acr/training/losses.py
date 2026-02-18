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
    alpha : float
        Balancing factor (default: ``1.0``).
    gamma : float
        Focusing parameter — higher values increase the effect
        (default: ``2.0``).
    reduction : str
        ``'mean'``, ``'sum'``, or ``'none'``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        inputs : Tensor, shape ``(N, C)``
            Raw logits (pre-softmax).
        targets : Tensor, shape ``(N,)``
            Ground-truth class indices.

        Returns
        -------
        Tensor
            Scalar loss (or per-sample if ``reduction='none'``).
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
