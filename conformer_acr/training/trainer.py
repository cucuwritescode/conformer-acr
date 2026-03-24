"""
conformer_acr.training.trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightweight training loop for the ConformerACR model.
Designed to be simple like Lightning but without the dependency,
and DDP-aware via :mod:`conformer_acr.utils.distributed`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast  #works on pytorch 1.x and 2.x
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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
    rank : int
        DDP rank (only rank 0 saves checkpoints and prints logs).
    use_ddp : bool
        Whether using DistributedDataParallel.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: str | torch.device = "cpu",
        checkpoint_dir: str | Path | None = None,
        rank: int = 0,
        use_ddp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.rank = rank
        self.use_ddp = use_ddp

        #AMP: mixed precision for ~2x speedup on V100 tensor cores
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

    def fit(
        self,
        train_loader: DataLoader,  #type: ignore[type-arg]
        val_loader: DataLoader | None = None,  #type: ignore[type-arg]
        epochs: int = 10,
        train_sampler: Optional[DistributedSampler] = None,
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
        train_sampler : DistributedSampler, optional
            DDP sampler (needs set_epoch for proper shuffling).

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
            #set epoch on sampler for proper DDP shuffling
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            log_msg = f"[Epoch {epoch:03d}/{epochs}] train_loss={train_loss:.4f}"

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                history["val_loss"].append(val_loss)
                log_msg += f" | val_loss={val_loss:.4f}"

            #only rank 0 prints and saves checkpoints
            if self.rank == 0:
                print(log_msg, flush=True)

                if self.checkpoint_dir is not None:
                    ckpt_path = self.save_checkpoint(epoch)
                    print(f"  -> Checkpoint saved: {ckpt_path}", flush=True)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:  # type: ignore[type-arg]
        """Run a single training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            #unpack batch dict and move tensors to device
            cqt = batch["cqt"].float().to(self.device)
            lengths = batch["lengths"].to(self.device)
            root = batch["root"].to(self.device)
            bass = batch["bass"].to(self.device)
            qual = batch["qual"].to(self.device)

            #create padding mask from lengths (bool to avoid AMP dtype issues)
            max_len = cqt.size(1)
            mask = (torch.arange(max_len, device=self.device).unsqueeze(0) >= lengths.unsqueeze(1)).bool()

            self.optimizer.zero_grad()

            #AMP: autocast forward pass to FP16
            with autocast(enabled=self.use_amp):
                out = self.model(cqt, mask=mask.bool())

                #compute loss for each head and sum
                loss_root = self.loss_fn(out["root"].transpose(1, 2), root)
                loss_qual = self.loss_fn(out["quality"].transpose(1, 2), qual)
                loss_bass = self.loss_fn(out["bass"].transpose(1, 2), bass)
                loss = loss_root + loss_qual + loss_bass

            #AMP: scaled backward + unscaled step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _validate(self, loader: DataLoader) -> float:  # type: ignore[type-arg]
        """Run validation. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                #unpack batch dict and move tensors to device
                cqt = batch["cqt"].float().to(self.device)
                lengths = batch["lengths"].to(self.device)
                root = batch["root"].to(self.device)
                bass = batch["bass"].to(self.device)
                qual = batch["qual"].to(self.device)

                #create padding mask from lengths (bool to avoid AMP dtype issues)
                max_len = cqt.size(1)
                mask = (torch.arange(max_len, device=self.device).unsqueeze(0) >= lengths.unsqueeze(1)).bool()

                #AMP: autocast inference
                with autocast(enabled=self.use_amp):
                    out = self.model(cqt, mask=mask.bool())

                    #compute loss for each head and sum
                    loss_root = self.loss_fn(out["root"].transpose(1, 2), root)
                    loss_qual = self.loss_fn(out["quality"].transpose(1, 2), qual)
                    loss_bass = self.loss_fn(out["bass"].transpose(1, 2), bass)
                    loss = loss_root + loss_qual + loss_bass

                total_loss += loss.item()
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
        tmp_path = path.with_suffix('.pt.tmp')

        #handle DDP model state_dict (remove 'module.' prefix for portability)
        if self.use_ddp:
            model_state = {k.replace("module.", ""): v for k, v in self.model.state_dict().items()}
        else:
            model_state = self.model.state_dict()

        #atomic write: save to tmp then rename (prevents corruption on crash)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            tmp_path,
        )
        os.replace(tmp_path, path)  #atomic on POSIX
        return path
