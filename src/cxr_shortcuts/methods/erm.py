# Empirical risk minimisation, the only training method used.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

def erm_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    if logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
    loss.backward()
    optimizer.step()
    return loss.item()

def erm_train_step_microbatched(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    micro_batch: Optional[int] = None,
) -> float:
    if micro_batch is None or micro_batch >= len(images):
        return erm_train_step(model, optimizer, images, labels, pos_weight=pos_weight)
    if micro_batch <= 0:
        raise ValueError(f"micro_batch must be positive, got {micro_batch}")
    model.train()
    optimizer.zero_grad()
    n = len(images)
    total = 0.0
    for start in range(0, n, micro_batch):
        chunk_x, chunk_y = images[start:start + micro_batch], labels[start:start + micro_batch]
        logits = model(chunk_x)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, chunk_y.float(), pos_weight=pos_weight) * (len(chunk_x) / n)
        loss.backward()
        total += loss.item()
    optimizer.step()
    return total
