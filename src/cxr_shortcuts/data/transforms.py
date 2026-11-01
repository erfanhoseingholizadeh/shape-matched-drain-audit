# Turns an edited uint8 film into the tensor an ImageNet-pretrained backbone expects (resize, three channels, normalisation).
# Run: library module, imported by the scripts

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def to_model_tensor(image: np.ndarray, size: int = 224) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"expected a grayscale HxW image, got shape {image.shape}")

    resized = np.array(Image.fromarray(image).resize((size, size), Image.BILINEAR))
    normalized = resized.astype(np.float32) / 255.0

    tensor = torch.from_numpy(normalized).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std
