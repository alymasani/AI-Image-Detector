import pytest
import torch
from torch import nn

from detector.models import MODEL_NAMES, SimpleCNN, build_model, build_resnet18


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def build_offline(name: str) -> nn.Module:
    # pretrained=False keeps the tests from downloading the ImageNet weights.
    return build_model(name, pretrained=False)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_outputs_two_logits_per_image(name):
    model = build_offline(name).eval()
    with torch.no_grad():
        logits = model(torch.randn(3, 3, 224, 224))
    assert logits.shape == (3, 2)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_parameter_is_trainable(name):
    assert all(p.requires_grad for p in build_offline(name).parameters())


def test_cnn_architecture():
    model = SimpleCNN()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert [c.out_channels for c in convs] == [32, 64, 128, 256]
    assert sum(isinstance(m, nn.MaxPool2d) for m in model.modules()) == 4
    assert model.classifier.in_features == 256 and model.classifier.out_features == 2


def test_resnet18_head_is_replaced():
    model = build_resnet18(pretrained=False)
    assert isinstance(model.fc, nn.Linear)
    assert (model.fc.in_features, model.fc.out_features) == (512, 2)


def test_cnn_can_overfit_a_small_batch():
    model = SimpleCNN()
    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 1] * 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05
    assert torch.equal(model(images).argmax(dim=1), labels)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="cnn"):
        build_model("vgg16")
