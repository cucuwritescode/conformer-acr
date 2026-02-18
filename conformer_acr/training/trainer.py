"""
conformer_acr.training.trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightweight training loop for the ConformerACR model.
Designed to be simple like Lightning but without the dependency,
and DDP-aware via :mod:`conformer_acr.utils.distributed`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    """Minimal training harness for ConformerACR.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    optimizer : Optimizer
        PyTorch optimiser instance.
    loss_fn : nn.Module
        Loss function (e.g. :class:`~conformer_acr.training.losses.FocalLoss`).
    device : str | torch.device
        Target device (``'cpu'``, ``'cuda'``, ``'cuda:0'``, …).
    checkpoint_dir : str | Path | None
        Directory to save checkpoints to. If *None*, no checkpoints are saved.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: str | torch.device = "cpu",
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def fit(
        self,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader | None = None,  # type: ignore[type-arg]
        epochs: int = 10,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        val_loader : DataLoader, optional
            Validation data (evaluated at the end of each epoch).
        epochs : int
            Number of training epochs.

        Returns
        -------
        dict[str, list[float]]
            History dict with keys ``'train_loss'`` and optionally
            ``'val_loss'``.
        """
        history: dict[str, list[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)

            if self.checkpoint_dir is not None:
                self.save_checkpoint(epoch)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:  # type: ignore[type-arg]
        """Run a single training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            # TODO: unpack batch dict and move tensors to self.device
            self.optimizer.zero_grad()
            # loss = self.loss_fn(...)
            # loss.backward()
            self.optimizer.step()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader) -> float:  # type: ignore[type-arg]
        """Run validation. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                # TODO: unpack batch dict and move tensors to self.device
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, epoch: int) -> Path:
        """Persist model + optimiser state.

        Parameters
        ----------
        epoch : int
            Current epoch number (used in the filename).

        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        assert self.checkpoint_dir is not None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        return path
