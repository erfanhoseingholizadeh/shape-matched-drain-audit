# Saves and restores training state so a run can resume after the job is killed.
# Run: library module, imported by the scripts

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
        },
        path,
    )

def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint["scheduler_state"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    rng_state = checkpoint["rng_state"]
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])

    return checkpoint["epoch"] + 1
