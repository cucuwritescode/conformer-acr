"""
conformer_acr.utils.distributed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Helpers for PyTorch Distributed Data Parallel (DDP) training,
with special handling for the **Bede** HPC cluster (NVLink topology,
``bede-mpirun`` launcher, Open-CE conda stacks).

All Bede/NVLink-specific ugliness lives here so the rest of the
library stays platform-agnostic.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def setup_ddp(
    backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """Initialise the DDP process group.

    On Bede, call this from within a ``bede-mpirun`` context
    so that ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, and
    ``MASTER_ADDR``/``MASTER_PORT`` are already set.

    Parameters
    ----------
    backend : str
        Communication backend (``'nccl'`` for GPU, ``'gloo'`` for CPU).
    init_method : str
        URL-style init method (default: ``'env://'``).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def cleanup_ddp() -> None:
    """Destroy the DDP process group (call at script exit)."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Return the global rank of this process (0 if not distributed)."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Return the total number of processes (1 if not distributed)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Return *True* on rank-0 (use for logging / checkpointing guards)."""
    return get_rank() == 0
