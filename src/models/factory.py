"""Select and construct a model from configuration, and report parameter counts."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .cnn import CustomCNN
from .resnet18 import build_resnet18

MODEL_DISPLAY_NAMES = {
    "cnn": "Custom CNN",
    "resnet18": "ResNet-18",
}


def build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Build a model whose forward maps ``N x 3 x 224 x 224`` images to ``N x 2`` logits."""
    name = model_cfg["name"]
    if name == "cnn":
        return CustomCNN(num_classes=2, dropout=float(model_cfg.get("dropout", 0.3)))
    if name == "resnet18":
        return build_resnet18(num_classes=2, pretrained=bool(model_cfg.get("pretrained", True)),
                              freeze_backbone=bool(model_cfg.get("freeze_backbone", False)))
    raise ValueError(f"Unknown model name {name!r}; expected one of {sorted(MODEL_DISPLAY_NAMES)}")


def initialization_source(model_cfg: Dict[str, Any]) -> str:
    name = model_cfg["name"]
    if name == "cnn":
        return "random (Kaiming/Xavier), trained from scratch"
    if name == "resnet18":
        if model_cfg.get("pretrained", True):
            return "Torchvision ResNet18_Weights.IMAGENET1K_V1, new 2-way fc" + (
                " (backbone frozen)" if model_cfg.get("freeze_backbone") else ", fully fine-tuned")
        return "random (Torchvision default init), trained from scratch"
    return "unknown"


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def display_name(model_cfg: Dict[str, Any]) -> str:
    base = MODEL_DISPLAY_NAMES.get(model_cfg["name"], model_cfg["name"])
    if model_cfg["name"] == "resnet18" and not model_cfg.get("pretrained", True):
        return base + " (scratch)"
    return base
