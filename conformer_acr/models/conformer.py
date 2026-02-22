"""
conformer_acr.models.conformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conformer encoder with three classification heads
(root, quality, bass) for automatic chord recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from conformer_acr.config import N_CQT_BINS, NUM_ROOTS, NUM_QUALITIES, NUM_BASS


class ConformerACR(nn.Module):
    """Conformer-based Automatic Chord Recognition model.

    Architecture
    ------------
    1. Linear projection of CQT input to ``d_model``
    2. Stack of Conformer encoder layers
    3. Three independent classification heads:
       - **root**    → 13-class softmax (C, C♯, …, B, N)
       - **quality** → 15-class softmax  (maj, min, dim, …, N)
       - **bass**    → 13-class softmax (12 pitch classes + no-bass)

    Parameters
    ----------
    d_model : int
        Embedding / hidden dimension of the Conformer layers.
    n_heads : int
        Number of attention heads in each Conformer block.
    n_layers : int
        Number of stacked Conformer encoder layers.
    input_dim : int
        Dimensionality of each input frame (default: ``N_CQT_BINS``).
    dropout : float
        Dropout probability applied throughout.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        input_dim: int = N_CQT_BINS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # ── Classification heads ────────────────────────────────────
        self.head_root = nn.Linear(d_model, NUM_ROOTS)
        self.head_quality = nn.Linear(d_model, NUM_QUALITIES)
        self.head_bass = nn.Linear(d_model, NUM_BASS)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Parameters
        ----------
        x : Tensor, shape ``(batch, time, input_dim)``
            CQT feature frames.
        mask : Tensor, optional
            Padding mask for the encoder.

        Returns
        -------
        dict[str, Tensor]
            Keys ``"root"``, ``"quality"``, ``"bass"`` mapping to logit
            tensors of shape ``(batch, time, num_classes)``.
        """
        h: torch.Tensor = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=mask)

        return {
            "root": self.head_root(h),
            "quality": self.head_quality(h),
            "bass": self.head_bass(h),
        }
