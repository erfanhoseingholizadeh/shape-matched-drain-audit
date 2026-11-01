# An ImageNet-pretrained ResNet-18 real-versus-edited detector, one member of the realism panel.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from cxr_shortcuts.edits.drain.realism_discriminator import (
    images_to_tensor_3ch,
    train_model,
)

class ResNet18Discriminator(nn.Module):

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)

def train_resnet_discriminator(
    real_images: Sequence[np.ndarray],
    edited_images: Sequence[np.ndarray],
    seed: int = 0,
    epochs: int = 30,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    pretrained: bool = True,
) -> nn.Module:
    return train_model(
        lambda: ResNet18Discriminator(pretrained=pretrained),
        real_images,
        edited_images,
        seed=seed,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        to_tensor_fn=images_to_tensor_3ch,
    )
