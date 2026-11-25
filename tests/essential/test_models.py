"""Model correctness: shapes, loss, gradients, updates, modes, parameter counts, checkpoint round-trip."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.cnn import CustomCNN
from src.models.factory import build_model, parameter_counts
from src.training.checkpointing import load_model_for_inference, save_checkpoint

CNN_PARAMS = 1_189_794       # exact count, computed programmatically and frozen here
RESNET18_2WAY_PARAMS = 11_177_538


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_cnn_output_shape_and_finite(batch):
    model = CustomCNN().eval()
    out = model(torch.randn(batch, 3, 224, 224))
    assert out.shape == (batch, 2) and torch.isfinite(out).all()


def test_cnn_has_four_blocks_no_residuals_and_exact_params():
    model = CustomCNN()
    assert len(model.features) == 4
    assert parameter_counts(model) == {"total": CNN_PARAMS, "trainable": CNN_PARAMS, "non_trainable": 0}
    assert not any("resnet" in type(m).__module__ for m in model.modules())
    assert not any(isinstance(m, (nn.Softmax, nn.Sigmoid)) for m in model.modules())


def test_loss_backward_gradients_and_update():
    torch.manual_seed(0)
    model = CustomCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])
    before = [p.detach().clone() for p in model.parameters()]
    loss = nn.CrossEntropyLoss()(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) > 0 for g in grads) == len(grads)
    opt.step()
    assert all(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_train_eval_mode_dropout():
    torch.manual_seed(0)
    model = CustomCNN(dropout=0.5)
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    feats = model.pool(model.features(x)).flatten(1).detach()
    head = model.classifier
    assert not torch.equal(head(feats), head(feats))  # dropout active in train mode


def test_resnet18_head_and_full_finetuning():
    model = build_model({"name": "resnet18", "pretrained": False, "freeze_backbone": False})
    assert model.fc.out_features == 2
    assert parameter_counts(model) == {"total": RESNET18_2WAY_PARAMS, "trainable": RESNET18_2WAY_PARAMS,
                                       "non_trainable": 0}
    assert model.eval()(torch.randn(2, 3, 224, 224)).shape == (2, 2)
    frozen = build_model({"name": "resnet18", "pretrained": False, "freeze_backbone": True})
    assert parameter_counts(frozen)["trainable"] == 512 * 2 + 2


def test_checkpoint_roundtrip_equivalence(tmp_path):
    torch.manual_seed(0)
    model = CustomCNN().eval()
    cfg = {"model": {"name": "cnn", "dropout": 0.3}, "training": {"seed": 0}}
    save_checkpoint(tmp_path / "m.pt", model, cfg, epoch=1)
    loaded, ckpt = load_model_for_inference(tmp_path / "m.pt")
    x = torch.randn(2, 3, 224, 224)
    assert torch.equal(model(x), loaded(x)) and ckpt["epoch"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
def test_cnn_cuda_forward():
    model = CustomCNN().cuda().eval()
    assert model(torch.randn(2, 3, 224, 224, device="cuda")).shape == (2, 2)
