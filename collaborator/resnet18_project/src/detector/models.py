"""The two detectors we compare: a small CNN trained from scratch and a pretrained ResNet-18.

Both take a batch of normalized 3x224x224 images and return logits of shape [B, 2], where
column 0 is "real" and column 1 is "fake".
"""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 2
MODEL_NAMES = ("cnn", "resnet18")


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """3x3 conv, ReLU, then 2x2 max pool, which halves the image size."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SimpleCNN(nn.Module):
    """Four conv blocks (32, 64, 128, 256 channels), global average pool, then a linear layer.

    224x224 input shrinks to 14x14 after the four pools, and the average pool turns that into
    one 256-number summary per image, so the model works on any input size.
    """

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.features(x))
        return self.classifier(x.flatten(1))


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """ResNet-18 with its 1000-class ImageNet head swapped for a 2-class one.

    The whole network stays trainable, so it is fully fine-tuned rather than used as a frozen
    feature extractor.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Build a model by name: "cnn" or "resnet18". `pretrained` only affects ResNet-18."""
    if name == "cnn":
        return SimpleCNN()
    if name == "resnet18":
        return build_resnet18(pretrained=pretrained)
    raise ValueError(f"Unknown model {name!r}, expected one of {MODEL_NAMES}")
