# Factory for the DenseNet-121, ResNet-50 and ViT-B/16 backbones with a multi-label head.
# Run: library module, imported by the scripts

from __future__ import annotations

from torch import nn
from torchvision import models

def build_backbone(name: str, num_classes: int = 1, pretrained: bool = True) -> nn.Module:
    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    raise ValueError(f"unknown backbone: {name!r} (only 'densenet121'/'resnet50'/'vit_b_16' are implemented)")
