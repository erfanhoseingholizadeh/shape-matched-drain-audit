# Real-versus-edited discriminators of the realism panel.
# Run: library module, imported by the scripts

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

class RealismDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features).squeeze(-1)

class DeepRealismDiscriminator(nn.Module):

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features).squeeze(-1)

def crop_around_local_mask(
    image: np.ndarray,
    offset: tuple[int, int],
    local_mask_shape: tuple[int, int],
    crop_size: int,
) -> np.ndarray:
    if crop_size > image.shape[0] or crop_size > image.shape[1]:
        raise ValueError(
            f"crop_size={crop_size} exceeds image shape {image.shape}"
        )

    row_offset, col_offset = offset
    mask_height, mask_width = local_mask_shape
    center_row = row_offset + mask_height // 2
    center_col = col_offset + mask_width // 2
    half = crop_size // 2

    top = max(0, min(center_row - half, image.shape[0] - crop_size))
    left = max(0, min(center_col - half, image.shape[1] - crop_size))
    return image[top : top + crop_size, left : left + crop_size]

def images_to_tensor(images: Sequence[np.ndarray]) -> torch.Tensor:
    stacked = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(stacked).unsqueeze(1)

def images_to_tensor_3ch(images: Sequence[np.ndarray]) -> torch.Tensor:
    return images_to_tensor(images).expand(-1, 3, -1, -1)

def train_model(
    model_factory: Callable[[], nn.Module],
    real_images: Sequence[np.ndarray],
    edited_images: Sequence[np.ndarray],
    seed: int = 0,
    epochs: int = 30,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    to_tensor_fn: Callable[[Sequence[np.ndarray]], torch.Tensor] = images_to_tensor,
    device: str = "cpu",
) -> nn.Module:
    torch.manual_seed(seed)
    model = model_factory()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    x = torch.cat([to_tensor_fn(real_images), to_tensor_fn(edited_images)], dim=0).to(device)
    y = torch.cat([torch.zeros(len(real_images)), torch.ones(len(edited_images))]).to(device)
    shuffle_generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(x.shape[0], generator=shuffle_generator)
        for start in range(0, x.shape[0], batch_size):
            batch_idx = permutation[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch_idx]), y[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model

def train_discriminator(
    real_images: Sequence[np.ndarray],
    edited_images: Sequence[np.ndarray],
    seed: int = 0,
    epochs: int = 30,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    device: str = "cpu",
) -> RealismDiscriminator:
    return train_model(
        RealismDiscriminator,
        real_images,
        edited_images,
        seed=seed,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
    )

def train_deep_discriminator(
    real_images: Sequence[np.ndarray],
    edited_images: Sequence[np.ndarray],
    seed: int = 0,
    epochs: int = 30,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    dropout: float = 0.3,
    device: str = "cpu",
) -> DeepRealismDiscriminator:
    return train_model(
        lambda: DeepRealismDiscriminator(dropout=dropout),
        real_images,
        edited_images,
        seed=seed,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
    )

def predict_scores(
    model: nn.Module,
    images: Sequence[np.ndarray],
    batch_size: int = 32,
    to_tensor_fn: Callable[[Sequence[np.ndarray]], torch.Tensor] = images_to_tensor,
) -> np.ndarray:
    device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = to_tensor_fn(images[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores)

def discriminator_auc(
    model: nn.Module,
    real_images: Sequence[np.ndarray],
    edited_images: Sequence[np.ndarray],
    to_tensor_fn: Callable[[Sequence[np.ndarray]], torch.Tensor] = images_to_tensor,
) -> float:
    scores = np.concatenate(
        [
            predict_scores(model, real_images, to_tensor_fn=to_tensor_fn),
            predict_scores(model, edited_images, to_tensor_fn=to_tensor_fn),
        ]
    )
    labels = np.concatenate([np.zeros(len(real_images)), np.ones(len(edited_images))])
    return float(roc_auc_score(labels, scores))
