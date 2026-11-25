"""Model A: custom four-block CNN trained from scratch (Group 38 original architecture).

Input ``N x 3 x 224 x 224`` -> four conv blocks (32, 64, 128, 256 channels; each two 3x3
conv-BN-ReLU layers followed by 2x2 max pooling, halving resolution 224->112->56->28->14)
-> global average pooling -> Linear(256, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 2).

Returns two raw logits (no softmax): ``nn.CrossEntropyLoss`` applies log-softmax internally,
which is numerically more stable than softmax followed by a log.

No pretrained weights, no residual connections, no torchvision model components are used.
Initialisation: Kaiming-normal (fan_out, ReLU gain) for conv layers, BatchNorm weight=1/bias=0,
Kaiming-uniform (PyTorch default style, ReLU gain) for the first linear layer and
Xavier-uniform for the final logits layer, all biases zero.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    """Two 3x3 conv -> BatchNorm -> ReLU layers then 2x2 max pooling.

    Conv layers have ``bias=False`` because the following BatchNorm adds its own learnable shift.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class CustomCNN(nn.Module):
    """Four-block CNN detector. ``forward(images) -> logits`` with shape ``N x 2``."""

    CHANNELS: List[int] = [32, 64, 128, 256]

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, in_channels: int = 3):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in self.CHANNELS:
            blocks.append(ConvBlock(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        hidden, logits = self.classifier[1], self.classifier[4]
        nn.init.kaiming_uniform_(hidden.weight, nonlinearity="relu")
        nn.init.zeros_(hidden.bias)
        nn.init.xavier_uniform_(logits.weight)
        nn.init.zeros_(logits.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def architecture_table(model: CustomCNN, image_size: int = 224) -> List[dict]:
    """Per-stage output shapes and exact parameter counts (computed, not estimated)."""
    rows = []
    x = torch.zeros(1, 3, image_size, image_size)
    model = model.eval()
    with torch.no_grad():
        for i, block in enumerate(model.features, start=1):
            x = block(x)
            rows.append({"stage": f"Block {i}", "layers": "2x[Conv3x3-BN-ReLU] + MaxPool2x2",
                         "output_shape": "x".join(map(str, x.shape[1:])),
                         "parameters": sum(p.numel() for p in block.parameters())})
        x = model.pool(x)
        rows.append({"stage": "Global avg pool", "layers": "AdaptiveAvgPool2d(1)",
                     "output_shape": "x".join(map(str, x.shape[1:])), "parameters": 0})
        out = model.classifier(x)
    rows.append({"stage": "Head", "layers": "Linear(256,64)-ReLU-Dropout(0.3)-Linear(64,2)",
                 "output_shape": "x".join(map(str, out.shape[1:])),
                 "parameters": sum(p.numel() for p in model.classifier.parameters())})
    rows.append({"stage": "Total", "layers": "", "output_shape": "2 logits",
                 "parameters": sum(p.numel() for p in model.parameters())})
    return rows
