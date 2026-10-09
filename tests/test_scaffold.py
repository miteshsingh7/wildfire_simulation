"""Initial scaffold test verifying environment, package structure, and torch."""

import sys

import torch

import wf052


def test_package_scaffold() -> None:
    """Verify package imports and version string."""
    assert wf052.__version__ == "0.1.0"
    assert sys.version_info >= (3, 11)


def test_pytorch_available() -> None:
    """Verify PyTorch is imported and functional without TensorFlow/Keras."""
    tensor = torch.zeros((2, 2))
    assert tensor.shape == (2, 2)
    assert "tensorflow" not in sys.modules
    assert "keras" not in sys.modules
