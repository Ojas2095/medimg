"""Model registry and factory functions for classification, segmentation, and detection.

Provides explicit architecture specification, input normalization parameters,
and unified model constructors for training and inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .models import ResNetUNet, UNet, classifier, fasterrcnn, retinanet


@dataclass(frozen=True)
class ModelSpec:
    """Specification of an architecture registered in the medimg workstation."""
    key: str
    task: str  # "classification", "segmentation", "detection"
    arch_name: str
    factory: Callable[..., nn.Module]
    default_checkpoint_name: str
    input_size: Tuple[int, int]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    num_classes: int
    description: str


# Standard ImageNet normalization for pretrained backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Flat normalization for raw models
IDENTITY_MEAN = (0.0, 0.0, 0.0)
IDENTITY_STD = (1.0, 1.0, 1.0)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "cls_resnet18": ModelSpec(
        key="cls_resnet18",
        task="classification",
        arch_name="resnet18",
        factory=lambda pretrained=False, **k: classifier("resnet18", pretrained=pretrained, n_out=1),
        default_checkpoint_name="cls_resnet18.pt",
        input_size=(128, 128),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        num_classes=1,
        description="ResNet-18 binary triage classifier with ImageNet initialization and BCE with pos_weight."
    ),
    "cls_resnet50": ModelSpec(
        key="cls_resnet50",
        task="classification",
        arch_name="resnet50",
        factory=lambda pretrained=False, **k: classifier("resnet50", pretrained=pretrained, n_out=1),
        default_checkpoint_name="cls_resnet50.pt",
        input_size=(128, 128),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        num_classes=1,
        description="ResNet-50 heavier binary classifier."
    ),
    "cls_googlenet": ModelSpec(
        key="cls_googlenet",
        task="classification",
        arch_name="googlenet",
        factory=lambda pretrained=False, **k: classifier("googlenet", pretrained=pretrained, n_out=1),
        default_checkpoint_name="cls_googlenet.pt",
        input_size=(128, 128),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        num_classes=1,
        description="GoogLeNet Inception classifier."
    ),
    "seg_resnetunet": ModelSpec(
        key="seg_resnetunet",
        task="segmentation",
        arch_name="resnet_unet",
        factory=lambda pretrained=False, **k: ResNetUNet(pretrained=pretrained),
        default_checkpoint_name="seg_resnetunet.pt",
        input_size=(128, 128),
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
        num_classes=1,
        description="U-Net style segmentor with ResNet-18 pretrained encoder and skip connections."
    ),
    "seg_unet": ModelSpec(
        key="seg_unet",
        task="segmentation",
        arch_name="unet",
        factory=lambda in_ch=3, base=16, **k: UNet(in_ch=in_ch, base=base),
        default_checkpoint_name="seg_unet.pt",
        input_size=(128, 128),
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
        num_classes=1,
        description="Standard from-scratch 4-depth U-Net encoder-decoder."
    ),
    "det_retinanet": ModelSpec(
        key="det_retinanet",
        task="detection",
        arch_name="retinanet",
        factory=lambda pretrained_backbone=False, **k: retinanet(num_classes=2, pretrained_backbone=pretrained_backbone),
        default_checkpoint_name="det_retinanet.pt",
        input_size=(128, 128),
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
        num_classes=2,
        description="RetinaNet with ResNet-50 FPN backbone and Focal Loss for lesion detection."
    ),
    "det_fasterrcnn": ModelSpec(
        key="det_fasterrcnn",
        task="detection",
        arch_name="fasterrcnn",
        factory=lambda pretrained_backbone=False, **k: fasterrcnn(num_classes=2, pretrained_backbone=pretrained_backbone),
        default_checkpoint_name="det_fasterrcnn.pt",
        input_size=(128, 128),
        mean=IDENTITY_MEAN,
        std=IDENTITY_STD,
        num_classes=2,
        description="Faster R-CNN two-stage detector."
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    """Retrieve model specification by registered key."""
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model key '{key}' not found in MODEL_REGISTRY. Available keys: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key]
