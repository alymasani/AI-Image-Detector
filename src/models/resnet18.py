"""Model B: ResNet-18 (Torchvision) with a two-logit head.

REFERENCE IMPLEMENTATION. The ResNet-18 model is assigned to another Group 38 member. At the
time this file was written no collaborator implementation existed in the repository, so this
reference keeps the project executable and fixes the shared interface (``logits = model(x)``,
``x: N x 3 x 224 x 224``, ``logits: N x 2``, labels 0=real / 1=AI). When the collaborator's code
or checkpoint arrives, integrate it through ``src/models/factory.py`` and record the provenance in
docs/ARCHITECTURE.md and docs/ATTRIBUTION.md; do not present this file as their work.

Weights: ``torchvision.models.ResNet18_Weights.IMAGENET1K_V1`` (He et al., 2016, trained on
ImageNet-1k). If ``pretrained=True`` and the weights cannot be downloaded/loaded, construction
FAILS rather than silently falling back to random initialisation.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """Create ResNet-18, replace ``fc`` (512->1000) by a fresh ``Linear(512, num_classes)``.

    With ``freeze_backbone=False`` (the proposal's protocol) every parameter is trainable.
    """
    if pretrained:
        try:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:  # network/cache failure must be visible
            raise RuntimeError(
                "Could not load Torchvision ImageNet weights for ResNet-18 (IMAGENET1K_V1). Check network access or "
                "pre-populate the torch hub cache ($TORCH_HOME). Refusing to fall back to random weights; set "
                "model.pretrained=false explicitly for the from-scratch experiment."
            ) from exc
    else:
        model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    for name, param in model.named_parameters():
        param.requires_grad = (not freeze_backbone) or name.startswith("fc.")
    return model
