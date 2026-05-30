import torch

from src.model import build_resnet18


def test_model_output_shape():

    model = build_resnet18(num_classes=2)

    x = torch.randn(1, 3, 224, 224)

    y = model(x)

    assert y.shape == (1, 2)