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


def test_seed_and_versions() -> None:
    """Verify reproducible seed utility and environment version collection."""
    from wf052.utils import get_environment_versions, seed_everything

    seed_everything(123)
    versions = get_environment_versions()
    assert "torch" in versions
    assert "numpy" in versions
    assert "python" in versions
