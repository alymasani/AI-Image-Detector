"""Model B: ImageNet-pretrained ResNet-18, fully fine-tuned, with a two-logit head.

COLLABORATOR'S MODEL. Integrated from the Group 38 ResNet-18 project, kept unmodified in
``collaborator/resnet18_project/src/detector/models.py`` (``build_resnet18``): Torchvision ResNet-18
with ``ResNet18_Weights.IMAGENET1K_V1``, the 1000-way ``fc`` replaced by ``nn.Linear(512, 2)``
(PyTorch's default Linear initialisation), and every layer left trainable (full fine-tuning). Its
label mapping (0 = real, 1 = fake/AI) and input contract (ImageNet-normalised 3x224x224) match this
repository, so no remapping is needed.

Two integration additions, both defaults-off with respect to the collaborator's behaviour:
* If ``pretrained=True`` and the weights cannot be loaded, construction FAILS (never a silent fallback
  to random weights); ``pretrained=False`` is the explicit switch for experiment E8.
* ``freeze_backbone`` (default False) for completeness; the proposal's protocol fine-tunes everything.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet-18 with its ImageNet head swapped for a ``num_classes``-way one (collaborator's construction)."""
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
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for name, param in model.named_parameters():
        param.requires_grad = (not freeze_backbone) or name.startswith("fc.")
    return model
